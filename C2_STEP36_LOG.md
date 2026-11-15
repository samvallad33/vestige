# C2 Step 36 log — Antaranga-bahiranga

Branch `perf/c2-step36-antaranga`, cut from `d121d063` (Step 3A). `git am -3` applied `steps7_10a` (5), then `algos44` (44), then `wins` (Steps 31, 32, 33, 34) with no conflicts. The hook order after the am is Step 32, then Step 33, then Step 34 (`apply_c2_step34_bertillonage(ix, &q.text, &mut final_turn_indices, &t_scene)`), then `maybe_dump_replay`. Step 36 is one commit on that tip.

Author: Sam Valladares `<samvalladares21@gmail.com>`.

Datasets, hashes checked before any bench:

- `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM-100K: 316 files under `BEAM/chats/100K`

## What the scores are

Verified in `rank_one` before any Step 36 edit.

- `t_own[ti]` is BM25F on `turn_own_bags` (uncarried), plus the contiguous query-bigram bonus. Scene pooling does not write it. This is the turn's own lexical match.
- `t_turn_raw[ti]` is that own score plus a fraction of bag-carry (`turn_bags` minus own), then the slot-shape, kelal, ring, and loci passes. It is the turn's self score before neighbour weights.
- `t_scene[ti]` starts as that self score plus neighbour weights (`prev_w * neighbour + next_w * neighbour` on the named-speaker path and the user/assistant path) and is then multiplied by the later scene passes. Seats do not rewrite it.
- `bridge_boost` is not added into `t_scene`. It is added only to the session rank score (`s_score + bridge_boost`), which is what Step 33 already uses.

External evidence on a turn is `t_scene - t_own`: neighbour carry and the later scene passes, relative to the turn's own lexical score. A turn is held mostly by that external score when the residual strictly exceeds `t_own` (`t_scene - t_own > t_own`). A turn is held by its own score when `t_own` is a strict majority of its own `t_scene` (`t_own > t_scene - t_own`).

## Dump fields

`write_one_replay_dump` (retrieval.rs), read before the edit:

- `final_ids` / `seated_ids` are the official turn order after the gatekeeper and every C2 seat. `rank` is the index in that list.
- `pre_ids` is that seated list first, then `t_scene`-descending turns with `t_scene > 0` until 50. `rank_pre` is the index in `pre_ids`. So `pre_ids[0]` is `final_ids[0]` whenever the seated list is non-empty. A literal "gold at pre_rank 0" equals "gold at final rank 0".
- Scene rank 0 is argmax `t_scene` over the pool (tie break lower `idx`), which is what `tools/c2_funnel.py` prints. There is no separate post-Kemeny snapshot: `rank1_gatekeeper_keep_rsb_head` runs before the C2 seats, and the dump sees the final seated list.
- `t_own`, `t_turn_raw`, `t_scene_sess`, `s_score`, `bridge_boost` are copies of the scores the seat already computed. `qtok_present` and `turn_ic_novel` read `TurnIx.body` and are dump-only. Ranking does not.

## Stage funnel, combined tip, before any Step 36 edit

Turn gold. Replay dumps of `/home/ubuntu/c2-bins/tip3134` (`d4e61d94`, binary sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`).

| Bench | n | corpus | pool@50 | pre_rank@5 | pre_rank @1 | final @5 | final @1 | scene @5 | scene @1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LME_S | 470 | 470 | 463 | 428 | 353 | 428 | 353 | 399 | 265 |
| LoCoMo | 1536 | 1535 | 1415 | 1127 | 808 | 1127 | 808 | 1110 | 708 |
| BEAM-100K | 355 | 355 | 328 | 239 | 160 | 238 | 160 | 221 | 124 |

LoCoMo 5-line funnel (scene rank 0 is argmax `t_scene`, not `rank_pre`):

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
  gold at final rank0 191
  gold in final top5 248
[LoCoMo Open-Domain] n=92
  gold in corpus     92
  gold in pool@50    63
  gold at scene rank0 14
  gold at final rank0 20
  gold in final top5 35
[LoCoMo Single-Hop] n=841
  gold in corpus     841
  gold in pool@50    810
  gold at scene rank0 440
  gold at final rank0 500
  gold in final top5 686
```

LME and LoCoMo `pre_rank@5` equal final @5. BEAM `pre_rank@5` is 239 and final @5 is 238 because one seated list is shorter than 5. One LoCoMo gold id is absent from that conversation's turns (corpus 1535/1536).

## Baseline scoreboard (measured, before the edit)

`bd2693e2` here is the algo-44 tree `5c246f3a` (steps 7–10A and the 44, no Steps 31–34), benched in this run. Its aggregates match the published bd2693e2 board on every metric below. The combined tip is `d4e61d94` (Steps 31+32+33+34). Per-question diff of the tip against that algo-44 bench: **0 losses** on every coordinate. The only changes are the Turn@1 gains listed in the prompt, all still hits:

- LME `e8a79c70`, `c4a1ceb8`, `ccb36322`
- LoCoMo `2:106`, `3:143`, `5:13`, `6:25`, `9:147`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36`
- BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0`

| | bd2693e2 | tip 31+32+33+34 |
| --- | --- | --- |
| LME Session @1/@5/@10, All@5 | 430/463/468, 424 | same |
| LME Turn @1/@5/@10, All@5 | 350/428/437, 314 | 353/428/437, 314 |
| LoCoMo Turn @1/@5/@10, Sess @1/@5 | 796/1127/1226, 1133/1416 | 808/1127/1226, 1133/1416 |
| BEAM Turn @1/@5/@10, All@5, Sess @1/@5 | 156/238/267, 118, 133/249 | 160/238/267, 118, 133/249 |

## What the misses are

On the tip, gold is already in the top 5 and not in slot 1 on 75 LME, 319 LoCoMo, and 78 BEAM questions. The literal reading (slot 1's external residual exceeds its own score; the unique max-`t_own` top-5 turn that covers slot 1's content anchors and whose own score is a majority of its scene takes slot 1) was scored on the replay rows and not committed: LME +0/−3, LoCoMo +21/−51, BEAM +1/−2. The losses are gold heads whose scene score is borrowed from a neighbour while a different top-5 turn repeats the query's surface tokens and outscores them on `t_own`. Promoting that echo undoes a seat that was right.

Same-session restrictions, donor exclusion, rarest-token guards, and "question yields to a declarative in the same session" did not produce a zero-loss gain. The cell that did: slot 1 has to be a question, the witness has to sit in a session whose `s_score + bridge_boost` is strictly higher than slot 1's, and the witness has to be held by its own score. A same-session echo shares slot 1's session score, so the session sightline refuses it. That is the same session score Step 33 uses, but Step 33 also requires the witness to be the unique maximum on both sightlines and to strictly cover the head's term ids, and it is named-speaker only. The four questions below are still misses after Step 33.

## Attempt 1 — shipped. WIN

Commit: the Step 36 commit on this branch (the only ranker change).

Files:

- `crates/vestige-core/src/algos_c2.rs` 3813–3932 (`step36_anchors`, `step36_posts`, `apply_c2_step36_antaranga`), test 7681–7747
- `crates/vestige-core/src/retrieval.rs` 6287–6299 (one call after `apply_c2_step34_bertillonage`, before `maybe_dump_replay`)

Rule. Top 5 only. Slot 1 yields when all of the following hold, and otherwise stays:

1. Slot 1 is a question (`TurnIx.is_question`).
2. Its external residual strictly exceeds its own score: `t_scene - t_own > t_own`.
3. A top-5 turn is not a question, its `t_own` strictly exceeds slot 1's `t_own`, and that `t_own` is a strict majority of its own `t_scene`.
4. It attests every content anchor slot 1 attests. An anchor is a query token that passes `is_content`, interned as its surface id and, when different, its stem. Attestation is `TurnIx.content` or `TurnIx.feats`. Turn bodies are not read. A token the index never interned is not an anchor.
5. Its session score `s_score + bridge_boost` strictly exceeds slot 1's.
6. Among turns that pass 3–5, the own score is unique. A tie seats nothing.

The witness is moved to slot 1. The top-5 set does not change, so Turn@5, Turn@10, and All@5 cannot move. The session list is not rewritten. `rsb_seat_behind` is not touched.

The replay proxy (dump `qtok_present`, which also probes the body) predicted LoCoMo `0:97`, `2:84`, `3:174` and BEAM `16:knowledge_update:1`. The release run, which attests from content and feature ids only, did not move `2:84` and did move `9:86`. The release per-question diff is the result.

Real run of `/home/ubuntu/c2-bins/step36` (sha256 `dd223baa27ac55950c3d9971203a1786dd32c4323a1b3b82c3434faf967eac9d`):

- LME_S Session 430/463/468, All@5 424, All@10 447. Turn 353/428/437, All@5 314. Every LME bit +0/−0.
- LoCoMo Turn **811**/1127/1226, Session 1133/1416. Single-Hop Turn@1 500 → 503. Multi-Hop, Temporal, and Open-Domain Turn@1 unchanged.
- BEAM-100K Turn **161**/238/267, All@5 118, All@10 138, Session 133/249/275. `knowledge_update` Turn@1 31 → 32.

Per-question bits vs the combined tip: LoCoMo Turn@1 +3/−0 (`0:97`, `3:174`, `9:86`); BEAM Turn@1 +1/−0 (`16:knowledge_update:1`). Every other bit on all three benches: +0/−0. LME Session drops: 0. The same four questions are the only Turn@1 gains against the algo-44 bench as well, on top of the sixteen tip gains, with zero losses.

Pool membership and `t_scene` are untouched, so pool@50 and scene rank stay on the table above. Final @5 stays 428 / 1127 / 238. Final @1 is LME 353, LoCoMo 811, BEAM 161.

The four seats, from the tip dumps:

- LoCoMo `0:97` ("What workshop did Caroline attend recently?"). Slot 1 `D8:1` is a question, own 0.05, scene 22.46. Gold `D4:13` owns 9.43 of a scene of 15.18 and sits in the higher session (0.975 > 0.939). The same-session echo `D8:2` has a higher own score (15.18) and the same session score as the question, so it is not the witness.
- LoCoMo `3:174` ("What does Nate rely on for cheer and joy?"). Slot 1 `D2:20` is a question, own 0.08, scene 25.93. Gold `D24:3` owns 13.42 of a scene of 23.64, session 1.122 > 0.965.
- LoCoMo `9:86` ("What sports activity is Calvin planning to try after the tour with Frank Ocean?"). Slot 1 `D25:1` is a question, own 42.83, scene 96.79. Gold `D9:15` owns 51.48 of a scene of 77.13, session 0.858 > 0.698.
- BEAM `16:knowledge_update:1` ("What is my total budget for holiday gifts this year?"). Slot 1 `314` is a question, own 108.15, scene 316.50. Gold `250` owns 113.88 of a scene of 213.83, session 1.444 > 1.023.

## Gates

- `cargo clippy -p vestige-core --all-targets -- -D warnings`: exit 0.
- `cargo check -p vestige-mcp`: exit 0.
- `cargo test -p vestige-core`: 564 passed, 0 failed. Includes `step36_antaranga_seats_own_majority_over_an_external_question`.
- `c2_hashcheck.py` on the release `retrieval_bench`: ranked ids 0/500, full responses 0/500.
- LME Session: 430/463/468 and All@5 424, identical to the tip and to bd2693e2. Per-question session coordinates: +0/−0.
