# C2 Step 41 — Queried-speaker reply seat (witness form)

Branch `perf/c2-step41-queried-speaker-seat`, cut from `d121d063` (`perf/step3a-panini-karaka`). Steps 7–10A, algos 1–44, wins through Step 40 applied with `git am -3`. Step 39 and Step 40 each conflicted with the W2-001 hook; both seats were kept, Step 39 then Step 40 then W2-001. `git am` minted new hashes, so the pack's Computer 2 tip `19a8e30aa1226706e7fffab863bf327149bb3269` is not an object in this clone. Replayed tip before this step: `1a0794b49e67e020f4540d7b727ce11e141d6097` (68 commits from `d121d063`). Author: Sam Valladares `<samvalladares21@gmail.com>`.

Hook order in `retrieval.rs` after the replay, confirmed before the Step 41 edit: Step 31 (assistant-attribution only), 32, 33, 34, 35, 36, 37, 38, Pool C session-key opener, Pool C absent session key, Pool B standing seat, buried lexical peak, anaphoric tail, unseated session peak, Step 39, Step 40, W2-001, `maybe_dump_replay`. Step 41 is an addition at two sites: immediately after `apply_c2_queried_speaker_seat` (pre-RSB, inside `!is_event_sequence && !assistant_attribution && !c2_text_preference`), and again after Step 40 and before W2-001 (inside `!is_event_sequence && !assistant_attribution`). Every earlier hook is still there. `rsb_seat_behind` is not called from this step.

Datasets, hashed before use: LoCoMo `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. LongMemEval_S `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` (parts concatenated, `xz -d`). BEAM-100K from `beam100k.tgz` sha256 `c45e091568a2a0cbc69157de6b9b86b0d62263a57052c0b4413800de3f0ff598`.

## Baseline (pack-43 tip, Steps 31–40 + Pools C/B/A + W2-001 together, before any Step 41 edit)

`retrieval_bench` sha256 `911ffb8951c0db69903dc060c340f8e1702dafc7133aafce59537d1464beaede`. This stack had not been measured together. Every listed flip is still a hit on the bit that step recorded. LME Session is 432/463/468, so the W2-001 session lifts are in this number.

| Bench | Metric | Combined tip |
| --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 432/463/468 |
| LME_S | Session All@5/All@10 | 424/447 |
| LME_S | Turn @1/@5/@10 | 355/429/438 |
| LME_S | Turn All@5 | 320 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 813/1136/1229 |
| LoCoMo | Session @1/@5 | 1138/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 162/240/269 |
| BEAM-100K | Turn All@5/All@10 | 122/142 |
| BEAM-100K | Session @1/@5/@10 | 140/249/275 |
| BEAM-100K | Session All@5/All@10 | 137/178 |

LME Turn@1 by type: single-session-user 55/64, multi-session 90/121, single-session-preference 12/30, temporal-reasoning 94/127, knowledge-update 62/72, single-session-assistant 42/56.

LoCoMo Turn@1 / Session@1: Multi-Hop 97/156, Temporal 192/243, Open-Domain 20/36, Single-Hop 504/703.

BEAM Turn@1: contradiction_resolution 32, event_ordering 7, information_extraction 20, instruction_following 7, knowledge_update 32, multi_session_reasoning 22, preference_following 7, summarization 6, temporal_reasoning 29.

Listed flips, all still hits on this binary (vectors are `c2_bench.py` bits):

- Step 31 LME `e8a79c70` Turn@1 `[1,1,1,1,1,1,1]`.
- Step 32 LME `c4a1ceb8` Turn@1; LoCoMo `5:13`, `2:106`, `3:143`, `9:147`, `6:25` Turn@1; BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1` Turn@1.
- Step 33 LoCoMo `4:28`, `4:44`, `8:124` Turn@1.
- Step 34 LME `ccb36322` Turn@1; LoCoMo `1:42`, `5:105`, `9:9`, `9:36` Turn@1; BEAM `19:multi_session_reasoning:0` Turn@1.
- Step 35 LME `gpt4_2ba83207` Turn@1.
- Step 36 LoCoMo `0:97`, `3:174`, `9:86` Turn@1; BEAM `16:knowledge_update:1` Turn@1.
- Step 37 LME `gpt4_fa19884d` Turn@1 `[1,1,1,1,1,1,1]`.
- Step 38 LoCoMo `6:58` Turn@1.
- Pool C LME `2b8f3739`, `d3ab962e`, `099778bb`, `dcfa8644`, `gpt4_fe651585` Turn All@5 = 1; `gpt4_1e4a8aec` `[1,1,1,0,1,1,1]`. BEAM `6:temporal_reasoning:0` All@5 = 1; `11:knowledge_update:0` and `20:knowledge_update:1` all stored bits 1; `19:multi_session_reasoning:0` All@5 = 1; `15:multi_session_reasoning:0` and `15:temporal_reasoning:1` `[0,1,1,0,0,0,0]` (Turn@5 and Turn@10).
- Pool B BEAM `1:instruction_following:0` Turn@1.
- Pool A LoCoMo `9:137`, `9:146`, `1:10`, `6:64` `[1,1,0,1,1]`; `9:113`, `9:131` `[0,1,0,1,1]`; `7:146` `[0,0,0,1,1]`.
- W2-001 Session@1: LME `5d3d2817`, `d6233ab6`; LoCoMo `2:54`, `3:88`, `4:110`, `4:177`, `8:94`; BEAM `2:information_extraction:1`, `7:information_extraction:0`, `7:knowledge_update:1`, `7:summarization:1`, `8:knowledge_update:1`, `10:knowledge_update:0`, `12:information_extraction:1`.
- Step 39 LoCoMo `6:15` `[1,1,0,1,1]`, `8:109` `[0,1,0,1,1]` (Turn@5).
- Step 40 LoCoMo `4:74` Turn@1 `[1,1,1,1,1]`.

## Stage funnel, printed before the ranker edit

Dump fields, from `write_one_replay_dump` in `retrieval.rs`:

- `pre_ids` is seated turns first, then `t_scene` descending while `t_scene > 0`, capped at 50. `pre_ids[0]` is `final_ids[0]` by construction. There is no separate post-Kemeny id list. `post_kemeny@5` is unavailable.
- `final_ids` / `seated_ids` is the official order after `rank1_gatekeeper_keep_rsb_head` and every later seat.
- `score_rows` is every indexed turn. Corpus membership is gold id ∈ `score_rows`. Ranking does not read `TurnIx.body`.
- `candidates[].role` is the speaker string. `qtok_present` is dump-only and is not `c2_term_ids`.

Any-gold hit:

| Bench | n | corpus | pool@50 | pre_rank@5 | pre@1 | final@1 | final@5 | scene@1 | named-speaker turns in pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | 1536 | 1535 | 1416 | 1136 | 813 | 813 | 1136 | 708 | 1536 |
| LoCoMo Multi-Hop | 282 | 282 | 243 | 158 | 97 | 97 | 158 | 72 | 282 |
| LoCoMo Temporal | 321 | 320 | 299 | 251 | 192 | 192 | 251 | 182 | 321 |
| LoCoMo Open-Domain | 92 | 92 | 63 | 35 | 20 | 20 | 35 | 14 | 92 |
| LoCoMo Single-Hop | 841 | 841 | 811 | 692 | 504 | 504 | 692 | 440 | 841 |
| BEAM-100K | 355 | 355 | 328 | 241 | 162 | 162 | 240 | 124 | 0 |
| LME_S | 470 | 470 | 463 | 429 | 355 | 355 | 429 | 265 | 0 |

`pre_ids[:1] == final[:1]` on all 1536 LoCoMo, all 355 BEAM, all 470 LME questions. `pre_ids[:5] == final[:5]` on 1535/1536 LoCoMo (the one miss is Open-Domain), 354/355 BEAM, 470/470 LME. BEAM pre top 5 is 241 and final top 5 is 240 because `pre_ids` fills past a seated list shorter than 5. Official Turn@5 is final@5, and it matches `c2_bench` (LoCoMo 1136, BEAM 240, LME 429). One LoCoMo gold id is absent from that conversation's turns (corpus 1535/1536). Pool length never exceeds 50. `score_rows` length equals `n_turns` on every question.

BEAM pool roles are only `user` and `assistant`. LME pool roles are only `user` and `assistant`. Slot 1 is a pool member on every question. A named-speaker seat cannot fire on either bench. The hunt moved to LoCoMo Turn@1, where every question has a named-speaker turn in the pool and 1136−813 = 323 questions already hold gold inside pre_rank@5.

## Attempt

One benched attempt. Shipped.

`apply_c2_step41_queried_speaker_seat` keeps the #15 seat and drops the question-head requirement. #15 is not edited. The same gates stay: text-level preference, summary request, WH-person, empty speaker, `user`/`assistant`, a head the query already names, non-positive scene score, conjunction (`c2_speaker_conjoined`), and a next turn outside the top-5 window (`C2_BULLAE_WINDOW`). The swap is `list.remove(slot); list.insert(0, cand)` with `slot >= 1` inside that window, so top-5 membership is unchanged and Session@5 cannot drop from this swap.

Two witnesses replace the question cut:

1. Head term-cover. For a head that is not a question, the reply's reported terms must strictly cover the head. Reported terms are `c2_term_ids` (own content or own features) minus that turn's own speaker-name ids and the stem ids of that name. Group-cover is not the object. A head that already attests every content term the reply attests stays. A sideways reply that posts a different term stays. Question heads still yield, subject to the temporal witness.
2. Temporal opening. `plan.intent.temporal` is not a blanket skip. The seat runs only when the reply carries a date or digit feature id the head lacks. Feature surfaces count when they start with `__DATE__`, `__MONTH__`, `__YEAR__`, `__YM__`, `__MD__`, `__SEASON__`, or `__SY__`, or when the surface contains an ASCII digit. Shared session-calendar features sit on both turns and do not count. No body read.

All four measured gains are `plan.intent.temporal` queries (`how long` is in `TEMPORAL_REST_CUES`; `when will` / `when did` are in `WHEN_INTERROGATIVE_CUES`). #15 returns before the seat on `plan.intent.temporal`, so these four were closed under the shipped rule. The control flow of Step 41 makes the swap itself the proof of an exclusive date/digit feature: without that feature the function returns before the swap.

The unit test locks the question-head yield, the declarative hold (the `0:94` shape), the empty-head yield, the sideways hold (the `9:147` shape), conjunction, WH-person, preference, summary, `user`/`assistant`, a reply outside the top 5, and the temporal digit arm. The first temporal fixture used "about 3 years". A lone digit of length 1 is not an index feature (`is_content` requires length greater than 1), so `c2_exclusive_date_digit` was false. The fixture was changed to "since 2019" before the release bench. That is a test fixture, not a ranker change.

Named questions the unwitnessed form was required to leave alone stayed at their baseline bits, all already Turn@1 hits: LoCoMo `0:94`, `4:165`, `7:116`, `9:80`, `9:88`, and also `1:78`, `8:136`, `9:147`.

## Shipped rule

**Files** (`git diff -U0` against `1a0794b4`):

- `crates/vestige-core/src/algos_c2.rs` lines 2785–2916 (`c2_reported_term_ids`, `c2_date_digit_surface`, `c2_exclusive_date_digit`, `apply_c2_step41_queried_speaker_seat`).
- `crates/vestige-core/src/algos_c2.rs` lines 10767–10925, unit test `step41_queried_speaker_reply_yields_only_to_a_strictly_richer_reply`.
- `crates/vestige-core/src/retrieval.rs` lines 6055–6065, pre-RSB call immediately after #15.
- `crates/vestige-core/src/retrieval.rs` lines 6393–6404, post-Step-40 call immediately before W2-001.

**Bench after the rule** (real `retrieval_bench` sha256 `8fea3e4702c388c5bedaadba1b9a1b8e2c5e86d4fcbb8b34bdca8892da6ff28d`):

| Bench | Before | After | Per-question |
| --- | --- | --- | --- |
| LME_S | 355/429/438 All@5 320; Session 432/463/468 All@5 424 All@10 447 | unchanged | 0 diffs / 470 |
| LoCoMo | Turn 813/1136/1229 Sess 1138/1416 | Turn **817**/1136/1229 Sess 1138/1416 | 4 gains, 0 losses / 1536 |
| BEAM-100K | Turn 162/240/269 All@5 122 All@10 142; Session 140/249/275 All@5 137 All@10 178 | unchanged | 0 diffs / 355 |

LoCoMo categories after: Multi-Hop Turn@1 97, Temporal **195** (was 192), Open-Domain 20, Single-Hop **505** (was 504). Session @1/@5 unchanged in every category. LME by-type and every BEAM category are unchanged. LME Session drops: 0. BEAM Session drops: 0. LoCoMo Session drops: 0.

**Gains.** All four are Turn@1 0→1 with Turn@5 already 1. Top-5 membership is unchanged. Session bits stay `[1,1]`. Slot 1 before is the previous turn in the same session; slot 1 after is the gold reply.

| question | category | before slot 1 | after slot 1 (gold) |
| --- | --- | --- | --- |
| `0:68` | Temporal | `D16:7` Caroline, question, "How long have you been into art?" | `D16:8` Melanie, "Seven years now" |
| `6:23` | Temporal | `D13:4` James, question, "When do you start?" | `D13:5` John, "I'm starting next month." |
| `7:3` | Temporal | `D1:5` Deborah, declarative, "she passed away a few years ago" | `D1:6` Jolene, "My mother also passed away last year." |
| `7:49` | Single-Hop | `D13:16` Deborah, question inside the turn, "How long have you been doing yoga and meditation?" | `D13:17` Jolene, "about 3 years now" |

`7:3` is the declarative case: the head has no `?`, so strict reported-term cover also had to hold. The other three heads are questions (`detect_question`), so the term-cover witness is not applied and the temporal date/digit witness is what opened the seat.

## Gates

- `cargo clippy -p vestige-core --all-targets -- -D warnings`: exit 0. First run compiled this diff (`Checking vestige-core`, `Finished dev profile in 14.18s`). Confirmation run `Finished dev profile in 0.11s`, exit 0.
- `cargo check -p vestige-mcp -j 2`: exit 0 (`Finished dev profile in 0.13s` on the already-checked tree).
- `cargo test -p vestige-core -j 2`: exit 0. Lib 579 passed, 0 failed, including `step41_queried_speaker_reply_yields_only_to_a_strictly_richer_reply`. `retrieval_bench` 0 tests. Doctests 3 passed, 10 ignored.
- `tools/c2_hashcheck.py` on the Step 41 binary, two independent LME passes (8 workers then 7): ranked ids 0/500, full response including f32 scores 0/500. Exit 0. 135.3s.
- LME Session: 0 drops (432/463/468, All@5 424, All@10 447).
- 0 per-question score losses vs the combined tip on LME, LoCoMo, and BEAM.
- LoCoMo Turn@1 strictly up (813 → 817).

## Still open

BEAM Turn@1 and Session@1, and LME Turn@1, are unchanged because the pool has no named speaker. Preference queries stay closed. `post_kemeny@5` is still not stored. The simulated +21 to +25 was headroom against an earlier list; on this tip the measured lift of the witness form is the four questions above.
