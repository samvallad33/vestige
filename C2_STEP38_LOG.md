# C2 Step 38 — Hermeneutic circle

Branch `perf/c2-step38-hermeneutic`, cut from `d121d063` (`perf/step3a-panini-karaka`). Steps 7–10A, algos 1–44, and Steps 31–36 applied cleanly with `git am -3`. Stack tip before this step: `b5d2e130` (Step 36 Antaranga-bahiranga). Author on the Step 38 commit: Sam Valladares.

Datasets: LoCoMo sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. LongMemEval_S sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` (the Hugging Face `xiaowu0162/longmemeval-cleaned` file `longmemeval_s_cleaned.json`; it matches the required hash). BEAM-100K from the attached `beam100k.tgz` (`BEAM/chats/100K`, 20 conversations).

## Baseline (Steps 31+32+33+34+35+36 together, before any Step 38 edit)

`retrieval_bench` at `b5d2e130`. This stack had not been measured together. The narrative floor said LME Turn@1 353; the combined run is 354. All four listed LME Turn@1 flips are hits (`e8a79c70`, `c4a1ceb8`, `ccb36322`, `gpt4_2ba83207`), which is 350 + 4. LoCoMo 811 and BEAM 161 match the sum of the listed flips. Every listed flip is still a Turn@1 hit. Session aggregates match bd2693e2, and Steps 31–36 do not rewrite `final_session_indices`.

| Bench | Metric | Combined tip |
| --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 |
| LME_S | Session All@5/All@10 | 424/447 |
| LME_S | Turn @1/@5/@10 | 354/428/437 |
| LME_S | Turn All@5 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 811/1127/1226 |
| LoCoMo | Session @1/@5 | 1133/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 161/238/267 |
| BEAM-100K | Turn All@5/All@10 | 118/138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 |

LME by type, Turn@1: single-session-user 55/64, multi-session 90/121, single-session-preference 12/30, temporal-reasoning 93/127, knowledge-update 62/72, single-session-assistant 42/56.

LoCoMo Turn@1: Multi-Hop 97, Temporal 191, Open-Domain 20, Single-Hop 503.

BEAM Turn@1: contradiction_resolution 32, event_ordering 7, information_extraction 20, instruction_following 6, knowledge_update 32, multi_session_reasoning 22, preference_following 7, summarization 6, temporal_reasoning 29.

## Stage funnel, printed before the ranker edit

Dump fields, from `write_one_replay_dump` in `retrieval.rs`:

- `pre_ids` is the pool: seated turns first, then `t_scene` descending while `t_scene > 0`, capped at 50. `pre_ids[0]` is `final_ids[0]` by construction, so dump pre-rank 0 is not an independent stage.
- `final_ids` / `seated_ids` is the official turn order after every seat, including Steps 31–36.
- `candidates[].t_scene` is the scene score copied at dump time. Seats reorder the list and do not rewrite it. Scene-rank 0 below is argmax `t_scene` over that pool (tie break: lower turn index).
- `qtok_present` is dump-only (body token or vocab/stem id in `turn.content`). Ranking does not read `TurnIx.body`.

Official `tools/c2_funnel.py` on LoCoMo, baseline binary:

| Category | n | corpus | pool@50 | scene rank 0 | final rank 0 | final top 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Multi-Hop | 282 | 282 | 243 | 72 | 97 | 158 |
| Temporal | 321 | 320 | 299 | 182 | 191 | 248 |
| Open-Domain | 92 | 92 | 63 | 14 | 20 | 35 |
| Single-Hop | 841 | 841 | 810 | 440 | 503 | 686 |

Same definition across all three benches (probe, baseline binary). Because `pre_ids` is seated-first, pre-rank @5 equals final top 5, and pre-rank @1 equals final @1.

| Bench | n | corpus | pool@50 | pre @1 | scene @1 | final @1 | final @5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LME_S | 470 | 470 | 463 | 354 | 265 | 354 | 428 |
| LoCoMo | 1536 | 1535 | 1415 | 811 | 708 | 811 | 1127 |
| BEAM-100K | 355 | 355 | 328 | 161 | 124 | 161 | 238 |

One LoCoMo temporal gold id is absent from that conversation's turns (corpus 1535/1536). That is in the baseline dump, not introduced here.

## Attempt 1 — shipped (WIN)

**Rule.** Among the top 5, content anchors are `step36_anchors` (closed class already dropped; unknown to the vocab is not an anchor). A turn posts an anchor from `turn.content` or `turn.feats` only. The session posts an anchor when any of its turns does. The part is the anchor set the turn itself posts. The rest is what that part does not post.

Slot 1 yields only when it is a question, its part is a nonempty proper subset, and exactly one other top-5 turn is that same part: not a question, a different session, a strictly stronger `t_own`, and a session that posts every rest anchor while slot 1's session misses at least one. Two such turns do not move the seat. Top-5 membership is unchanged. No new constant (`C2_BULLAE_WINDOW`). Speaker is not read. `rsb_seat_behind` is not touched. The session list is not rewritten.

This is not Step 36. Antaranga requires the question's external residual to exceed its own score and the witness's own score to be a majority of its scene. On the winning question the head residual does not exceed its own score (103.6 vs 103.8) and the witness's own score is not a majority of its scene, so Step 36 does not fire.

**Files.** `crates/vestige-core/src/algos_c2.rs` lines 4084–4172 (`apply_c2_step38_hermeneutic`) and lines 8191–8299 (unit test). `crates/vestige-core/src/retrieval.rs` lines 6307–6312 (call after Step 36, before `maybe_dump_replay`).

**Probe, before the edit, equal-part cross-session completion** (dump `qtok_present`, not a bench score). Fires with losses unless the witness's own score is strictly stronger:

| Gate on that base | Probe gains | Probe losses |
| --- | --- | --- |
| none (equal part, other session covers the rest, head session does not) | 2 | 5 |
| `t_own` strictly greater | 1 | 0 |
| head is a question, witness is not, `t_own` strictly greater | 1 | 0 |

The losses without the own-score guard included LME `6d550036` and `031748ae`, LoCoMo `4:30`, `4:75`, `5:100`, `7:2`, and BEAM `4:contradiction_resolution:0` and `3:knowledge_update:1`. Those predicates were not committed.

**Bench after the commit's rule** (real `retrieval_bench`, not the probe):

| Bench | Before | After | Per-question |
| --- | --- | --- | --- |
| LME_S | 354/428/437 All@5 314; Session 430/463/468 All@5 424 All@10 447 | unchanged | 0 diffs / 470 |
| LoCoMo | Turn 811/1127/1226 Sess 1133/1416 | Turn **812**/1127/1226 Sess 1133/1416 | 1 gain, 0 losses / 1536 |
| BEAM-100K | Turn 161/238/267 All@5 118 All@10 138; Session 133/249/275 | unchanged | 0 diffs / 355 |

LoCoMo categories after: Multi-Hop Turn@1 97, Temporal **192** (was 191), Open-Domain 20, Single-Hop 503. Session @1/@5 unchanged in every category.

**Gain.** LoCoMo `6:58` (Temporal) Turn@1 0→1. Question: "When did James try Cyberpunk 2077 game?" Gold `D28:27`. Before, slot 1 was `D26:10` (a question; posts Cyberpunk / 2077 / game; session D26 never posts "try"). After, slot 1 is `D28:27` ("yesterday I tried Cyberpunk 2077"), same part, own score 112.47 > 103.63, and session D28 also posts "try" and "James".

**Score-neutral reorder.** LoCoMo `5:31` ("What has Andrew done with his dogs?") moves `D25:5` to slot 2 and `D13:13` to slot 1. Neither turn is gold, and the score vector is unchanged (`[0,1,0,1,1]` before and after). Top-5 membership is unchanged. Not a scored loss.

**Gates.**

- `cargo clippy -p vestige-core --all-targets -- -D warnings` clean.
- `cargo check -p vestige-mcp` clean.
- `cargo test -p vestige-core`: 566 passed, 0 failed.
- `--hash-check`: ranked ids 0/500, full response 0/500.
- LME Session: 0 drops (430/463/468, All@5 424, All@10 447).
- 0 per-question score losses vs the combined tip on LME, LoCoMo, and BEAM.
- LoCoMo Turn@1 strictly up (811 → 812).
