# C2 Step 39 — Capture-recapture

Branch `perf/c2-step39-capture-recapture`, cut from `d121d063` (`perf/step3a-panini-karaka`). Steps 7–10A, algos 1–44, Steps 31–38, Pool C, Pool B, Pool A, and the `#[test]` attribute fix applied with `git am -3` and zero conflicts. Hook order matches the pack: Step 32, 33, 34, 35, 36, 37, 38, Pool C opener, Pool C absent key, Pool B, buried lexical peak, anaphoric tail, unseated session peak, then `maybe_dump_replay`. Replayed tip before this step is `f8570b9a` (author Sam Valladares). Computer 2's tip `a2a7f2b9` is not in this clone; the tree is the clean replay of that series. Author on the Step 39 commit: Sam Valladares.

Datasets: LoCoMo sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. LongMemEval_S sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. The three uploaded zip parts each hold one `lme_s.json.xz.partN`; concatenating the xz pieces and decompressing yields `longmemeval_s_cleaned.json` with that digest, byte-identical to the file the benches scored. Concatenating the zip files themselves is not one archive (each zip's central directory sits at its own end, so `unzip` only sees part 3). BEAM-100K from `beam100k.tgz` (`BEAM/chats/100K`, 20 conversations, 355 questions).

## Baseline (Steps 31–38 + Pools C, B, A together, before any Step 39 edit)

`retrieval_bench` at `f8570b9a`. This stack had not been measured together. The run is the sum of the separately measured cells, not a new interaction loss.

| Bench | Metric | Combined tip |
| --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 |
| LME_S | Session All@5/All@10 | 424/447 |
| LME_S | Turn @1/@5/@10 | 355/429/438 |
| LME_S | Turn All@5 | 320 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 812/1134/1229 |
| LoCoMo | Session @1/@5 | 1133/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 162/240/269 |
| BEAM-100K | Turn All@5/All@10 | 122/142 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 |

LME by type, Turn@1: single-session-user 55/64, multi-session 90/121, single-session-preference 12/30, temporal-reasoning 94/127, knowledge-update 62/72, single-session-assistant 42/56.

LoCoMo Turn@1: Multi-Hop 97, Temporal 192, Open-Domain 20, Single-Hop 503. Turn@5: Multi-Hop 158, Temporal 250, Open-Domain 35, Single-Hop 691.

BEAM Turn@1: contradiction_resolution 32, event_ordering 7, information_extraction 20, instruction_following 7, knowledge_update 32, multi_session_reasoning 22, preference_following 7, summarization 6, temporal_reasoning 29.

Every listed flip is still a hit on the bit that step recorded:

- LME Turn@1: `e8a79c70`, `c4a1ceb8`, `ccb36322`, `gpt4_2ba83207`, `gpt4_fa19884d`
- LME Turn All@5 (and Turn@5/@10 where that was the flip): `2b8f3739`, `d3ab962e`, `099778bb`, `gpt4_1e4a8aec`, `dcfa8644`, `gpt4_fe651585`
- LoCoMo Turn@1: `5:13`, `2:106`, `3:143`, `9:147`, `6:25`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36`, `0:97`, `3:174`, `9:86`, `6:58`
- LoCoMo Turn@5 and Turn@10: `9:137`, `7:146`, `9:113`, `9:131`, `9:146`, `1:10`, `6:64`
- BEAM Turn@1: `1:instruction_following:0`, `9:temporal_reasoning:1`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0`, `11:knowledge_update:0`, `16:knowledge_update:1`, `20:knowledge_update:1`
- BEAM Turn@5/@10: `15:multi_session_reasoning:0`, `15:temporal_reasoning:1`
- BEAM All@5: `6:temporal_reasoning:0`, `11:knowledge_update:0`, `19:multi_session_reasoning:0`, `20:knowledge_update:1`

## Stage funnel, printed before the ranker edit

Dump fields, from `write_one_replay_dump` in `retrieval.rs`:

- `pre_ids` is the pool: seated turns first, then `t_scene` descending while `t_scene > 0`, capped at 50. `pre_ids[0]` is `final_ids[0]` by construction, so dump pre-rank 0 is not an independent stage.
- `final_ids` / `seated_ids` is the official turn order after every seat, including Steps 31–38 and the pool seats.
- `candidates[].t_scene` is the scene score copied at dump time. Seats reorder the list and do not rewrite it. Scene-rank 0 below is argmax `t_scene` over that pool (tie break: lower turn id).
- `score_rows` is `[id, session_id, turn_pos, t_scene_bits, t_own_bits, t_turn_raw_bits, flags]`. `session_rows` is `[session_id, s_score_bits, bridge_bits]`. Ranking does not read `TurnIx.body`. The body string is filled only while `VESTIGE_REPLAY_DUMP` is set. A paired no-dump / dump run of all 355 BEAM questions returned identical turn ids.
- Kemeny-Young is `rank1_gatekeeper`. It reorders the window before the C2 seats. The dump has no separate post-Kemeny id list. Final top-5 is after that gatekeeper and every later seat. Because `pre_ids` is seated-first, pre-rank @5 equals final top 5, and pre-rank @1 equals final @1.

| Bench | n | corpus | pool@50 | pre @1 | scene @1 | final @1 | final @5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LME_S | 470 | 470 | 463 | 355 | 266 | 355 | 429 |
| LoCoMo | 1536 | 1535 | 1416 | 812 | 708 | 812 | 1134 |
| BEAM-100K | 355 | 355 | 328 | 162 | 124 | 162 | 240 |

One LoCoMo temporal gold id is absent from that conversation's turns (corpus 1535/1536). That is in the baseline dump.

LoCoMo by category: Multi-Hop 282 corpus 282 pool 243 scene0 72 final0 97 final5 158. Temporal 321 / 320 / 299 / 182 / 192 / 250. Open-Domain 92 / 92 / 63 / 14 / 20 / 35. Single-Hop 841 / 841 / 811 / 440 / 503 / 691.

## Attempt 1 — not shipped

Turn-level Petersen on the top 5 of `t_own` and the top 5 of the session sightline. `N = n1 * n2 / m` was 5–9 on almost every multi-gold question, because a session-constant score does not share turns with the lexical top. Giving the second session those seats, including swaps that pulled turns in from outside the top 10, lost on the replay probe (LoCoMo Turn@5 losses in the dozens, BEAM All@5 losses). No scalar was changed. The capture was the wrong population. Not compiled into a bench binary.

## Attempt 2 — shipped (WIN)

**Rule.** The two captures are sessions. The lexical ranking orders turns with `t_own > 0` by `t_own`, then turn id. The session ranking orders turns with a positive session sightline (`s_score + bridge_boost`) or a positive `t_scene` by that sightline, then `t_scene`, then turn id. Each capture is the first three distinct sessions. `m` is the overlap. `N = n1 * n2 / m`. `m = 0` does not fire. Identical session sets do not fire. `N` outside 2..=4 does not fire.

The second session is the highest session sightline in the union other than slot 0's session. It is filled up to `min(N, 2)` seats in the top 5. One call moves one seat: its unique best non-question already in ranks 5..9, whose own score strictly beats the question it swaps with. That question must keep a same-session sibling in the top 5. Slot 0 stays. The top-10 set stays. The session list is not rewritten. Speaker is not read. `TurnIx.body` is not read. `rsb_seat_behind` is not touched. No new constant: the window is the existing top 5, and the capture width is three sessions.

**Files.** `crates/vestige-core/src/algos_c2.rs` lines 4910–5060 (`apply_c2_step39_capture_recapture`) and lines 5080–5110 (unit test). `crates/vestige-core/src/retrieval.rs` lines 6362–6376 (call after Pool A, before `maybe_dump_replay`).

**Bench** (real `retrieval_bench`, dump off):

| Bench | Before | After | Per-question |
| --- | --- | --- | --- |
| LME_S | 355/429/438 All@5 320; Session 430/463/468 All@5 424 All@10 447 | unchanged | 0 diffs / 470 |
| LoCoMo | Turn 812/1134/1229 Sess 1133/1416 | Turn 812/**1136**/1229 Sess 1133/1416 | 2 gains, 0 losses / 1536 |
| BEAM-100K | Turn 162/240/269 All@5 122 All@10 142; Session 133/249/275 | unchanged | 0 diffs / 355 |

LoCoMo categories after: Multi-Hop Turn@5 158, Temporal **251** (was 250), Open-Domain 35, Single-Hop **692** (was 691). Turn@1 and Session @1/@5 unchanged in every category. BEAM every category cell unchanged. LME every type cell unchanged. LME Session drops: 0.

**Gains.**

- LoCoMo `6:15` (Temporal) Turn@5 0→1. "When did James start playing Civilization VI?" Gold `D8:29`. It sat at rank 5. It swapped with `D22:4`, a question whose own score is 0. Slot 0 stayed `D22:5`. `D22` keeps other turns in the top 5. Vector `[1,1,0,0,1]` → `[1,1,0,1,1]`.
- LoCoMo `8:109` (Single-Hop) Turn@5 0→1. "What fun activity did Evan mention doing in July 2023?" Gold `D8:26` (own 24.53) sat at rank 8 and swapped with `D4:14`, a question (own 8.60). Slot 0 stayed `D4:16`. Vector `[0,1,0,0,1]` → `[0,1,0,1,1]`.

**Score-neutral reorder.** LoCoMo `0:13` (Multi-Hop, "What career path has Caroline decided to persue?") moves `D1:11` from rank 9 to rank 4 and `D4:10` the other way. Both evidence turns (`D4:13`, `D1:11`) are then in the top 5. `D4:13` was already slot 0, so Turn@1 and Turn@5 were already hits. The published bit vector is unchanged. Not a scored loss.

**Gates.**

- `cargo clippy -p vestige-core --all-targets -- -D warnings` clean.
- `cargo check -p vestige-mcp` clean.
- `cargo test -p vestige-core`: 574 passed, 0 failed.
- `--hash-check`: ranked ids 0/500, full response 0/500.
- LME Session: 0 drops (430/463/468, All@5 424, All@10 447).
- 0 per-question score losses vs the combined tip on LME, LoCoMo, and BEAM.
- LoCoMo Turn@5 strictly up (1134 → 1136).

The 90% line is not this step's result. LoCoMo Turn@1 is 812 against a need of 1383, and BEAM All@5 is 122 against a need of 320. Those gaps are larger than one top-5 seat. This commit is the clean uplift the gate asked for.
