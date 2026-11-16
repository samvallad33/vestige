# C2 Step 42 — Witness-margin session swap (W2-006)

Branch `perf/c2-step42-witness-margin-swap`, cut from `d121d063be3f22c59e40b6f39d7a14c905569a43`. Pack-43 applied with `git am -3` (`steps7_10a`, `algos44`, `wins`). Steps 39 and 40 conflicted at the retrieval hook. Resolution kept every earlier hook and this order: Steps 31–38, Pool C opener, Pool C absent key, Pool B, buried lexical peak, anaphoric tail, unseated session peak, Step 39, Step 40, W2-001, then `maybe_dump_replay`. Replay tip before W2-006: `31c17e6a189b81a7d11ffbf4005758f25aceebb7`. Author: Sam Valladares.

Datasets, sha256 checked before use:

- BEAM-100K archive `c45e091568a2a0cbc69157de6b9b86b0d62263a57052c0b4413800de3f0ff598`
- LoCoMo `locomo10.json` `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- LongMemEval_S `longmemeval_s_cleaned.json` `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` (parts concatenated, then `xz -d`)

## RULE 2 funnel, printed before the ranker edit

Verified in `write_one_replay_dump` (`retrieval.rs`):

- `pre_ids` is the pool. It is the seated turn list first, then turns with `t_scene > 0` in scene order, capped at 50. `rank_pre` is that pool index. It is not a snapshot from before Kemeny-Young.
- `final_ids` / `seated_ids` is the official turn order after `rank1_gatekeeper` (Kemeny-Young) and every later seat, including Steps 31–40.
- Because `pre_ids` starts with the seated list, `pre_ids[:5]` equals `final_ids[:5]` whenever at least five turns are seated. One BEAM question has gold in the scene fill inside the first five pool slots and not in the shorter seated list, so pool pre-rank @5 is 241 and final @5 is 240.
- The dump has no separate post-Kemeny id list. `rank1_gatekeeper` reorders the window and later seats overwrite that order. The post-Kemeny column below is gold in `final_ids[:5]`: after the gatekeeper and every later seat.
- `score_rows` is `[id, session_id, turn_pos, t_scene_bits, t_own_bits, t_turn_raw_bits, flags]` for every indexed turn. `session_rows` is `[session_id, s_score_bits, bridge_bits]`. Ranking does not read `TurnIx.body`. The body string is filled only while `VESTIGE_REPLAY_DUMP` is set. Flag bit 0 is `is_question`.
- `@1` below is gold in `final_ids[:1]`. It matches Turn@1 on all three benches. `scene@1` is argmax `t_scene` over the pool, tie broken by the lower turn index. It is not a dumped rank.

| Bench | n | corpus | pool@50 | pre_rank@5 | post_kemeny@5 (final@5) | @1 | scene@1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LME_S | 470 | 470 | 463 | 429 | 429 | 355 | 265 |
| LoCoMo | 1536 | 1535 | 1416 | 1136 | 1136 | 813 | 708 |
| BEAM-100K | 355 | 355 | 328 | 241 | 240 | 162 | 124 |

One LoCoMo gold turn id is absent from that conversation's indexed turns (corpus 1535/1536).

## Baseline scoreboard (pack-43 tip, before W2-006)

| Bench | Metric | Tip |
| --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 432/463/468 |
| LME_S | Session All@5/All@10 | 424/447 |
| LME_S | Turn @1/@5/@10 | 355/429/438 |
| LME_S | Turn All@5/All@10 | 320/346 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 813/1136/1229 |
| LoCoMo | Session @1/@5/@10 | 1138/1416/1488 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 162/240/269 |
| BEAM-100K | Turn All@5/All@10 | 122/142 |
| BEAM-100K | Session @1/@5/@10 | 140/249/275 |

## Attempt 1 — simulated, not shipped

Raw arms, evaluated on the tip dump (`score_rows` best `t_scene` per session, sessions 2–3 only). Not compiled into a bench binary.

- Arm A, integer 5:4 (`best_c * 4 >= best_inc * 5`), ties between equal challengers stay. BEAM Session@1 140→146 (+6 / 0 losses). LoCoMo 1138→1141 (+43 / −40). LME 432→434 (+3 / −1). The LME loss is `dad224aa` (knowledge-update).
- Arm B, strict 1:1, user/assistant only (`!c2_named_session`). BEAM 140→158 (+23 / −5). LoCoMo unchanged (named-speaker gate). LME 432→431 (+4 / −5), including `dad224aa` and `gpt4_2ba83207`.

Failed the 0-loss gate. LoCoMo Session@1 is already far ahead of Turn@1 (1138 vs 813), so a scene-margin promotion of sessions 2–3 demotes correct named-speaker heads. No scalar was changed.

## Attempt 2 — shipped (WIN)

**Rule.** One hook, `apply_c2_w2006_witness_margin`, after `apply_c2_w2001_session_head` and before `maybe_dump_replay`. The session list is only reordered. The turn list is not written.

Both arms look at sessions 2–3 (indices 1 and 2). The challenger must be the session of the rank-1 turn, and it must hold at least as many top-5 turns as the incumbent. Equal best scores stay.

- Arm A: `best_c * 4 >= best_inc * 5`, widened through f64 so the integer margin is exact for finite f32.
- Arm B: strict `best_c > best_inc`, and the incumbent session does not already cover every query cue posted on the rank-1 turn (`query_cues` / `turn_query_cues` / `covers_cues`). Empty head cues do not fire arm B.

Structural guards required by attempt 1, applied before either arm:

- The incumbent session is user/assistant. Named-speaker transcripts (LoCoMo) are not reordered. Same dialogue-type flag as `c2_named_session`. Speaker is a field, not a word list.
- The incumbent's best turn is not a question (`TurnIx.is_question`). This is what removes `dad224aa`: its gold head's peak is a question, and the rank-1 turn lives in another session.

`TurnIx.body` is not read. `rsb_seat_behind` is not touched. No new map.

**Files.** `crates/vestige-core/src/algos_c2.rs` lines 5464–5584 (`w2006_named_session`, `w2006_five_four`, `w2006_top5`, `apply_c2_w2006_witness_margin`) and lines 10759–10830 (unit test). `crates/vestige-core/src/retrieval.rs` lines 6399–6410 (call after W2-001, before `maybe_dump_replay`).

**Bench** (release `retrieval_bench`, dump off for the scored response; a paired dump of the same run matched turn ids and session ids, 0 mismatches):

| Bench | Before | After | Per-question |
| --- | --- | --- | --- |
| LME_S | Session 432/463/468 All@5 424 All@10 447; Turn 355/429/438 All@5 320 All@10 346 | Session **434**/463/468, everything else unchanged | 2 gains, 0 losses / 470 |
| LoCoMo | Turn 813/1136/1229 Sess 1138/1416/1488 | unchanged | 0 diffs / 1536 |
| BEAM-100K | Turn 162/240/269 All@5 122 All@10 142; Session 140/249/275 All@5 137 All@10 178 | Session **148**/249/275, everything else unchanged | 8 gains, 0 losses / 355 |

**Gains.** All are Session@1 0→1. No other bit moves.

- BEAM `1:summarization:1`, `4:contradiction_resolution:0`, `8:temporal_reasoning:0`, `10:knowledge_update:1`, `10:temporal_reasoning:1`, `11:instruction_following:1`, `16:preference_following:1`, `17:temporal_reasoning:0`
- LME `b5ef892d` (multi-session), `gpt4_d6585ce9` (temporal-reasoning)

**Gates.**

- `cargo clippy -p vestige-core --all-targets -- -D warnings` exit 0. Tail: `Finished dev profile [optimized + debuginfo] target(s) in 12.80s`.
- `cargo check -p vestige-mcp` exit 0. Tail: `Finished dev profile [optimized + debuginfo] target(s) in 23.88s`.
- `cargo test -p vestige-core` exit 0. `579 passed; 0 failed` (lib), `0 passed` (second target), `3 passed; 0 failed; 10 ignored`.
- `--hash-check` via `tools/c2_hashcheck.py`: ranked ids `0/500`, full response including f32 scores `0/500`.
- LME session drops: 0. Session@5 and Session@10 unchanged. No question lost Session@1.

## Not benched

A looser guard set (rank-1 session, question-peak refusal, strict 1:1, no top-5 count, no cue witness) simulated BEAM Session@1 +11 and LME +2 with 0 losses on the tip dump. It was not compiled. The shipped rule keeps the top-5 count and the arm-B cue witness from the brief. The bench number is +8 / +2, not +11.
