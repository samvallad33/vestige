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

### Why the rest of the census is still open

61 multi-session user openers and 49 temporal user openers are in pool@50 and not in the top 5. Attempt 1 could see many of them and lost `ac031881` doing it. Attempt 2 keeps that assistant and therefore also keeps the assistants that mention a query word the opener never posted. Those openers stay outside the top 5. The next structure has to tell a hapax answer (the "LIV" token, df 1, not on the opener) from an elaboration that only adds words the rest of the haystack already uses. That is not a re-tune of this cover.
