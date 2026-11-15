# Pool C log — session-key opener

Branch `perf/c2-pool-lme`, cut from `d121d063` (`perf/step3a-panini-karaka`). `git am -3` applied `steps7_10a` (5), `algos44` (44), then `wins` (Steps 31, 32, 33, 34) with no conflicts. Hook order at the tip, verified in `rank_one`: `apply_c2_step32_trial_balance`, then `apply_c2_step33_triangulation`, then `apply_c2_step34_bertillonage`, then `maybe_dump_replay`.

Author on the Pool C commit: Sam Valladares `<samvalladares21@gmail.com>`.

Datasets, hashes checked before any bench:

- `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM-100K: 21 conversation directories under `BEAM/chats/100K`, 355 scored questions

## Baseline scoreboard (measured, before any Pool C edit)

Binary sha256 `c05b1d053b72300041d088f005332c07755df38bf56b72b5591c395462c98c5e`. This is the first joint measurement of Steps 31+32+33+34.

| | Measured |
|---|---|
| LME Session @1/@5/@10, All@5/All@10 | 430/463/468, 424/447 |
| LME Turn @1/@5/@10, All@5 | 353/428/437, 314 |
| LoCoMo Turn @1/@5/@10, Session @1/@5 | 808/1127/1226, 1133/1416 |
| BEAM Turn @1/@5/@10, All@5/All@10, Session @1/@5/@10 | 160/238/267, 118/138, 133/249/275 |

Every listed flip is still a Turn@1 hit: LME `e8a79c70`, `c4a1ceb8`, `ccb36322`; LoCoMo `5:13`, `2:106`, `3:143`, `9:147`, `6:25`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36`; BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0`.

Aggregates other than Turn@1 match the published bd2693e2 board exactly. Turn@1 is bd2693e2 plus those flips and nothing else in the totals (LME 350+3, LoCoMo 796+12, BEAM 156+4). LME and BEAM do not run Step 33 (named-speaker only). LoCoMo Turn@5/@10 and both session metrics match bd2693e2, so Steps 32, 33, and 34 did not move them.

### Stage funnel, before any Pool C edit

Any-gold. `pre_rank 0` is `final rank 0` by the dump construction in `write_one_replay_dump`. Scene rank 0 is the argmax of `t_scene` and is not that number.

```
[LME_S] n=470
  gold in corpus       470
  gold in pool@50      463
  gold at pre_rank 0   353
  gold at final rank 0 353
  gold in final top5   428
[LoCoMo] n=1536
  gold in corpus       1532   (every gold id is a turn id; 1535 have at least one)
  gold in pool@50      1415
  gold at pre_rank 0   808
  gold at final rank 0 808
  gold in final top5   1127
[BEAM-100K] n=355
  gold in corpus       355
  gold in pool@50      328
  gold at pre_rank 0   160
  gold at final rank 0 160
  gold in final top5   238
```

Full category tables and the gold-turn stage counts are in `POOL_MISS_CENSUS_C.md`.

## Attempt 1 — rejected. Union cover

Structure. The user turn at session position 0 replaces a same-session assistant in slots 2–5 when the assistant's query groups remain covered by the opener or by any other turn that stays in the top 5. Digit on the assistant and not on the opener keeps the assistant. Slot 1 frozen. Named-speaker corpora, text-level preference, summaries, event sequences, and assistant-attribution questions skipped.

Not committed. The union is the wrong witness: a neighbour can cover the group the opener lacks, and the assistant that actually answers is the one removed.

Per-question: 11 gains, 1 loss.

Loss: LME `ac031881` (single-session-assistant) Turn@5, Turn@10, and Turn All@5. The gold is assistant turn `answer_sharegpt_GYqnAhC_190#5` ("LIV"). The seat replaced it with the session opener. `assistant_attribution` was false on that question. The opener does not post `designation` / `jumpsuit` / `records`; a different user turn still in the top 5 does, so the union let the swap through.

Gains that this loss disqualifies: LME All@5 on `2b8f3739`, `d3ab962e`, `099778bb` (multi-session) and `gpt4_1e4a8aec`, `dcfa8644`, `gpt4_fe651585` (temporal, the last of these also Turn@5 and Turn@10); BEAM `6:temporal_reasoning:0` All@5, `11:knowledge_update:0` All@5, `15:multi_session_reasoning:0` Turn@5/Turn@10, `15:temporal_reasoning:1` Turn@5/Turn@10, `20:knowledge_update:1` All@5.

Headline if it had been legal: LME Turn All@5 314 → 319 with Turn@5 428 → 428 except the assistant loss (net Turn@5 428 → 427, Turn@10 437 → 436). BEAM Turn@5 238 → 240, All@5 118 → 121. LoCoMo unchanged. One loss fails the gate.

## Attempt 2 — shipped. The opener is the witness

This commit, parent `57b4e023` (Step 34). Binary sha256 `58a129f4309549385bf176976a555842643dd2b635650bfc708aecc00d51cc15`.

Same geometry as attempt 1, with a different cover. The opener itself must attest every query group the assistant attests. A neighbour in the top 5 does not count. That is the session key, not a committee.

`ac031881` does not fire: the opener misses `designation`, `jumpsuit`, and `records`, which the gold assistant posts. The unit test `pool_c_session_opener_replaces_same_session_assistant_echo` locks that case, the digit case, named speakers, and text-level preference.

### Files

- `crates/vestige-core/src/algos_c2.rs` 3809–3903 (`apply_c2_pool_c_session_opener_seat`), test 7652–7792
- `crates/vestige-core/src/retrieval.rs` 6287–6290 (one call after `apply_c2_step34_bertillonage`, before `maybe_dump_replay`)

Does not read `TurnIx.body`. Does not write `rsb_seat_behind`. Does not edit `rungs.rs` or `benchmarks/real_engine/`. `DetHashMap` / `DetHashSet` only where the existing group helpers already use them. No new scalar.

### Funnel on the shipped rule

`t_scene` is unchanged, so scene rank 0 is unchanged. Final ranks moved on two questions.

LME `gpt4_fe651585`, replayed on both binaries:

- Baseline: both golds in pool@50. Final rank 0 is a gold. Top 5 holds one gold (`answer_65600ff6_1#8`). The other gold, session-2 opener `answer_65600ff6_2#0`, is in the pool and not in the top 5.
- Pool C: slot 1 unchanged. The opener takes the seat of assistant `answer_65600ff6_2#9`. Both golds are in the top 5. pre_rank 0 is still the same gold.

Question-level any-gold top 5 and rank 0 are unchanged on all three benches (LME 428 / 353, LoCoMo 1127 / 808, BEAM 238 / 160). All@5 moves because the second gold enters.

### Scoreboard

| | Baseline | Attempt 2 | Delta |
|---|---:|---:|---:|
| LME Turn All@5 | 314 | 315 | +1 |
| LME Turn @1/@5/@10 | 353/428/437 | same | 0 |
| LME Session @1/@5/@10 | 430/463/468 | same | 0 |
| LoCoMo Turn and Session | 808/1127/1226, 1133/1416 | same | 0 |
| BEAM Turn @1/@5/@10 | 160/238/267 | same | 0 |
| BEAM All@5 / All@10 | 118 / 138 | 119 / 139 | +1 / +1 |
| BEAM Session @1/@5/@10 | 133/249/275 | same | 0 |

Gains: LME `gpt4_fe651585` Turn All@5; BEAM `20:knowledge_update:1` All@5. Losses: 0. Temporal-reasoning All@5 74 → 75. Multi-session All@5 stays 53. Detail in `SCOREBOARD_POOL_C.md` and `POOL_C_PERQ_DIFF.md`.

### Why attempt 2 left the census open

61 multi-session user openers and 49 temporal user openers are in pool@50 and not in the top 5. The opener-itself cover refuses an assistant that posts any query group the opener does not. Most of those assistants do. The rejected union would have seated the opener, and it also removed `ac031881`.

## Attempt 3 — shipped. Union cover, two sessions already in the window

Same seat as attempt 2. Two changes, both structural.

The opener or any other turn that stays in the top 5 may attest a query group the assistant attests. The opener still has to post at least one group of its own. A digit on the assistant that the opener lacks still keeps the assistant.

The window must already contain two sessions. `ac031881` is one session end to end (top 5 is `answer_sharegpt_GYqnAhC_190` only). The seat returns before the loop. Every attempt-1 gain already had two or more sessions in the baseline top 5 (LME 2, 3, or 4; every BEAM gain 5). The gate deletes fires. It does not add any.

Parent commit `82c45088c489cc72399c3031fa39896bb266964b`. Binary sha256 `4c764e0ed91d1ba6a37a48c50d366f0723c26d7eb23412767ca52d8758df9b05`.

### Files

- `crates/vestige-core/src/algos_c2.rs` 3809–3923, test 7672–7888
- `crates/vestige-core/src/retrieval.rs` 6287–6291

### Funnel

`t_scene` is unchanged. Final ranks moved on 11 questions, all gains.

`2b8f3739` (multi-session, three gold openers). Baseline top 5 already held two of them and the session-3 assistant `answer_23759615_3#7`. Attempt 3 puts session-3 opener `answer_23759615_3#0` in that seat. All three golds are in the top 5. Slot 1 stays.

`gpt4_1e4a8aec` (temporal). Baseline top 5 held no gold. The gold opener `answer_16bd5ea6_2#0` takes the seat of assistant `answer_16bd5ea6_2#5`. Turn@5, Turn@10, and All@5 flip. Two sessions were already in the window.

`ac031881` top 5 is byte-identical to the baseline, gold assistant `answer_sharegpt_GYqnAhC_190#5` included.

### Scoreboard versus the Step 31–34 tip

| | Tip | Attempt 3 | Delta |
|---|---:|---:|---:|
| LME Turn @1/@5/@10 | 353/428/437 | 353/429/438 | 0 / +1 / +1 |
| LME Turn All@5 | 314 | 320 | +6 |
| LME Session @1/@5/@10, All@5 | 430/463/468, 424 | same | 0 |
| LoCoMo Turn @1/@5/@10 | 808/1127/1226 | same | 0 |
| LoCoMo Session @1/@5 | 1133/1416 | same | 0 |
| BEAM Turn @1/@5/@10 | 160/238/267 | 160/240/269 | 0 / +2 / +2 |
| BEAM All@5 / All@10 | 118 / 138 | 121 / 141 | +3 / +3 |
| BEAM Session @1/@5/@10 | 133/249/275 | same | 0 |

LME by category, cells that moved: multi-session All@5 53 → 56. Temporal-reasoning Turn@5 111 → 112, Turn@10 116 → 117, All@5 74 → 77. Single-session-assistant Turn@5 stays 52.

BEAM cells that moved: knowledge_update All@5 24 → 26, All@10 27 → 29. multi_session_reasoning Turn@5 34 → 35, Turn@10 36 → 37. temporal_reasoning Turn@5 37 → 38, Turn@10 39 → 40, All@5 24 → 25, All@10 27 → 28.

Gains, 11, losses 0, including against attempt 2. `ac031881` stays a hit. Detail in `SCOREBOARD_POOL_C.md` and `POOL_C_PERQ_DIFF.md`.

### What attempt 3 still left open

The two-session gate is what made the union legal. It also means a gold opener whose session is the only session in the top 5 never enters. 31 of the 61 multi-session openers in pool@50 and outside the top 5 have no same-session turn in the top 5 at all, so the same-session seat cannot see them.

## Attempt 4 — shipped. Absent session key

A user opener whose session is outside the top 5 takes an assistant from a non-dominant session. The assistant's own session must keep another turn, and that turn must attest every query group the assistant attests. The dominant session (most turns in the window, earliest slot on a tie) is not a donor. The opener must post a query group and have `t_scene > 0`. A digit on the assistant that the opener lacks keeps the assistant. One-session windows do not fire. Same named-speaker, preference, summary, event, and assistant-attribution skips as the other seat.

On the baseline census, 49 of the 52 LME gold assistants already in the top 5 are single-session-assistant. 47 of those sit in the dominant session, which this seat does not touch. The two that do not (`1903aded`, `b759caee`) did not lose a bit.

Parent `ca63ca034a620b85628f0f5068b3bbc0a40823a7`. Binary sha256 `97880b59ffa49ffeb9b9c06b51a6ae98db81add9c82e947bcb77d4cfa87411a4`.

### Files

- `crates/vestige-core/src/algos_c2.rs` 3925–4060 (`apply_c2_pool_c_absent_session_key`), test 8028–8113
- `crates/vestige-core/src/retrieval.rs` 6292–6295, after the same-session seat

### Scoreboard versus attempt 3

LME is question-identical (Turn 353/429/438, All@5 320, session 430/463/468). LoCoMo is question-identical. BEAM adds one gain and no losses.

| | Attempt 3 | Attempt 4 | Delta |
|---|---:|---:|---:|
| BEAM All@5 / All@10 | 121 / 141 | 122 / 142 | +1 / +1 |
| BEAM multi_session_reasoning All@5 / All@10 | 4 / 7 | 5 / 8 | +1 / +1 |
| BEAM Turn @1/@5/@10 | 160/240/269 | same | 0 |
| LME and LoCoMo | attempt 3 | same | 0 |

The new question is BEAM `19:multi_session_reasoning:0`, All@5 0 → 1. Versus the Step 31–34 tip that is the sixth BEAM gain, and the LME six are unchanged. Losses versus the tip and versus attempt 3: 0.

### Still open

LME multi-session All@5 is 56/121. Temporal-reasoning All@5 is 77/127. This seat did not move either. The openers it could have seated on LME were either not redundant with a staying companion, or the donor assistant was in the dominant session and was left there.
