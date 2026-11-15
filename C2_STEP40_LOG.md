# C2 Step 40 — Cue validity

Branch `perf/c2-step40-cue-validity`, cut from `d121d063` (`perf/step3a-panini-karaka`). Steps 7–10A, algos 1–44, Steps 31–38, Pool C, Pool B, Pool A, and the `#[test]` attribute fix applied cleanly with `git am -3`. No hook conflicts. Replayed tip before this step: `9d95cc42a6778cc5e044c6a244c2fe12c714c915`. The pack's Computer 2 tip `a2a7f2b92d03fb4a738fa300607f57837a071e7c` is that `#[test]` commit; `git am` assigns a new hash. Author: Sam Valladares `<samvalladares21@gmail.com>`.

Hook order in `retrieval.rs` after the replay, confirmed before the Step 40 edit: Step 31 (assistant-attribution only), 32, 33, 34, 35, 36, 37, 38, Pool C session-key opener, Pool C absent session key, Pool B standing seat, buried lexical peak, anaphoric tail, unseated session peak, `maybe_dump_replay` with dump-only `score_rows` / `session_rows`.

Datasets: LoCoMo sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. LongMemEval_S sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` (`xiaowu0162/longmemeval-cleaned` `longmemeval_s_cleaned.json`; it matches the required hash). BEAM-100K from `beam100k.tgz` (`BEAM/chats/100K`, 20 conversations).

## Baseline (Steps 31–38 + Pools C/B/A together, before any Step 40 edit)

`retrieval_bench` sha256 `dd449987923b653de2423edb495553da47d9ee005a5a5d2ea6b875be4ff1afbf`. This stack had not been measured together. Every listed flip is still a hit. LME Session matches the published floor, so Session drops are 0.

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
| BEAM-100K | Session All@5/All@10 | 137/178 |

Against the separately measured cells, the joint run is the sum of the non-overlapping flips, plus one BEAM Turn@1 the separate sheets did not add (162, not 161). That extra hit is already in this baseline. It is not a Step 40 change.

LME Turn@1 by type: single-session-user 55/64, multi-session 90/121, single-session-preference 12/30, temporal-reasoning 94/127, knowledge-update 62/72, single-session-assistant 42/56.

LoCoMo Turn@1: Multi-Hop 97, Temporal 192, Open-Domain 20, Single-Hop 503.

BEAM Turn@1: contradiction_resolution 32, event_ordering 7, information_extraction 20, instruction_following 7, knowledge_update 32, multi_session_reasoning 22, preference_following 7, summarization 6, temporal_reasoning 29.

Listed flips, all still hits on this binary:

- Step 31 LME `e8a79c70` Turn@1.
- Step 32 LME `c4a1ceb8` Turn@1; LoCoMo `5:13`, `2:106`, `3:143`, `9:147`, `6:25` Turn@1; BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1` Turn@1.
- Step 33 LoCoMo `4:28`, `4:44`, `8:124` Turn@1.
- Step 34 LME `ccb36322` Turn@1; LoCoMo `1:42`, `5:105`, `9:9`, `9:36` Turn@1; BEAM `19:multi_session_reasoning:0` Turn@1.
- Step 35 LME `gpt4_2ba83207` Turn@1.
- Step 36 LoCoMo `0:97`, `3:174`, `9:86` Turn@1; BEAM `16:knowledge_update:1` Turn@1.
- Step 37 LME `gpt4_fa19884d` Turn@1 `[1,1,1,1,1,1,1]`.
- Step 38 LoCoMo `6:58` Turn@1.
- Pool C LME `2b8f3739`, `d3ab962e`, `099778bb`, `dcfa8644`, `gpt4_fe651585` All@5 = 1; `gpt4_1e4a8aec` `[1,1,1,0,1,1,1]`. BEAM `6:temporal_reasoning:0` All@5 = 1; `11:knowledge_update:0` and `20:knowledge_update:1` all stored bits 1; `19:multi_session_reasoning:0` All@5 = 1; `15:multi_session_reasoning:0` and `15:temporal_reasoning:1` `[0,1,1,0,0,0,0]`.
- Pool B BEAM `1:instruction_following:0` Turn@1.
- Pool A LoCoMo `9:137`, `9:146`, `1:10`, `6:64` `[1,1,0,1,1]`; `9:113`, `9:131` `[0,1,0,1,1]`; `7:146` `[0,0,0,1,1]`.

## Stage funnel, printed before the ranker edit

Dump fields, from `write_one_replay_dump` in `retrieval.rs`:

- `pre_ids` is the pool: seated turns first, then `t_scene` descending while `t_scene > 0`, capped at 50. `pre_ids[0]` is `final_ids[0]` by construction, so pre-rank 0 is not an independent stage.
- `final_ids` / `seated_ids` is the official turn order after every seat, including Steps 31–38 and the pool seats.
- `candidates[].t_scene` is the scene score copied at dump time. Seats reorder the list and do not rewrite it. Scene-rank 0 below is argmax `t_scene` over that pool (tie break: lower turn index).
- `score_rows[0]` is every indexed turn id. Corpus membership is gold id ∈ `score_rows`. Ranking does not read `TurnIx.body`. `qtok_present` is dump-only.

| Bench | n | corpus | pool@50 | pre_rank 0 | scene rank 0 | final rank 0 | pre top 5 | final top 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LME_S | 470 | 470 | 463 | 355 | 265 | 355 | 429 | 429 |
| LoCoMo | 1536 | 1535 | 1416 | 812 | 708 | 812 | 1134 | 1134 |
| BEAM-100K | 355 | 355 | 328 | 162 | 124 | 162 | 241 | 240 |

One LoCoMo gold id is absent from that conversation's turns (corpus 1535/1536). BEAM pre top 5 is 241 and final top 5 is 240 because `pre_ids` fills past a seated list shorter than 5; pre-rank 0 still equals final rank 0.

## Attempts

All four attempts were scored on the same pre-edit dumps (`anchor_posts` via `step36_posts`, `anchor_sdf` via `step40_anchor_sdf`). A fire swaps inside the top 5, so only Turn@1 can change. Attempts 1–3 were not committed.

**Attempt 1 — inverse session-df, or separation `1 - ln(sdf)/ln(N)`, over the whole top 5.** Validity mass is the sum of per-cue weights. The unique strict maximum above slot 1 takes the seat. Inverse-df, no scene guard: LME +10/−89, LoCoMo +53/−203, BEAM +10/−79. Separation index, no scene guard: LME +10/−82, LoCoMo +44/−175, BEAM +11/−73. Failed. A whole-list reweight is the score multiplier the spec rules out, and the losses confirm it.

**Attempt 2 — same weights, but only a candidate whose scene score is greater than or bit-identical to slot 1.** Inverse-df: LME +2/−40, LoCoMo +1/−23, BEAM +8/−35. Separation index: LME +2/−39, LoCoMo +1/−21, BEAM +9/−32. Failed. Moderately rare cues still outvote a correct head.

**Attempt 3 — validity 1 only for a cue posted in exactly one session, scene `>=` slot 1.** LME +1/−3 (`gpt4_21adecb5` vs `afdc33df`, `9ee3ecd6`, `2a1811e2`), LoCoMo +1/−8 (`4:74` vs eight Single-Hop/Temporal losses), BEAM +1/−0 (`11:multi_session_reasoning:1`). The BEAM gain does not survive the other benches. Widening the tie past bit identity, including "slot 1's cues are in every session", picks up LoCoMo `2:82` as a loss. Failed.

**Attempt 4 — shipped.** Validity is 1 iff session-df is 1, else 0. The near-tie is bit-identical `t_scene` inside the existing top-5 window (`C2_BULLAE_WINDOW`). The unique member of that tie with a strictly higher validity sum takes slot 1. Two winners stay put. Fifty-one questions have some top-5 turn with slot 1's scene bits. This predicate fires once: LoCoMo `4:74`, a gain. Zero losses, zero neutral reorders, on the dumps. The release bench matches that delta.

This is not Step 32 (rarest turn-idf lexeme, opener or echo arm), Step 36 (external residual vs own-score majority), Step 37 (adjacent same-speaker restatement), or Step 38 (question head, same anchor part, other session covers the rest). On `4:74` slot 1 is not a question, so Step 38 cannot fire. Speaker is not read. `rsb_seat_behind` is not touched. The session list is not rewritten.

## Shipped rule

**Files.** `crates/vestige-core/src/algos_c2.rs` lines 3994–4006 (`step40_anchor_sdf`) and 4924–4997 (`apply_c2_step40_cue_validity`); unit test lines 9930–10002. `crates/vestige-core/src/retrieval.rs` lines 6362–6366 (call after Pool A, before `maybe_dump_replay`) and dump-only `anchor_sdf` / `anchor_posts` at 7554–7557, 7667, 7782. The dump fields are not read by ranking.

**Bench after the commit's rule** (real `retrieval_bench`, not the probe):

| Bench | Before | After | Per-question |
| --- | --- | --- | --- |
| LME_S | 355/429/438 All@5 320; Session 430/463/468 All@5 424 All@10 447 | unchanged | 0 diffs / 470 |
| LoCoMo | Turn 812/1134/1229 Sess 1133/1416 | Turn **813**/1134/1229 Sess 1133/1416 | 1 gain, 0 losses / 1536 |
| BEAM-100K | Turn 162/240/269 All@5 122 All@10 142; Session 133/249/275 | unchanged | 0 diffs / 355 |

LoCoMo categories after: Multi-Hop Turn@1 97, Temporal 192, Open-Domain 20, Single-Hop **504** (was 503). Session @1/@5 unchanged in every category.

**Gain.** LoCoMo `4:74` (Single-Hop) Turn@1 0→1. Question: "What aspects of the Harry Potter universe will be discussed in John's fan project collaborations?" Gold `D1:16`. Before, slot 1 was `D1:3` ("That's great! I just signed with a new team - excited for the season!"). After, slot 1 is `D1:16` ("We'll be discussing various aspects of the Harry Potter universe, like characters, spells, and magical creatures."). Scene bits match (`297.7453308105469`). Anchor session-df on this question, in content-token order, is `[2, 13, 13, 1, 29, 5, 2, 1]` (aspects, harry, potter, universe, john, fan, project, collaborations). `D1:3` posts only the session-df 29 anchor. `D1:16` posts the session-df 1 anchor `universe` and does not post the session-df 29 anchor. Validity 0 yields to validity 1. Own scores 0.237 vs 67.097; the seat does not compare them. Top-5 membership is unchanged.

## Gates

- `cargo clippy -p vestige-core --all-targets -- -D warnings` clean.
- `cargo check -p vestige-mcp` clean.
- `cargo test -p vestige-core`: 574 passed, 0 failed.
- `--hash-check`: ranked ids 0/500, full response 0/500.
- LME Session: 0 drops (430/463/468, All@5 424, All@10 447).
- 0 per-question score losses vs the combined tip on LME, LoCoMo, and BEAM.
- LoCoMo Turn@1 strictly up (812 → 813).
