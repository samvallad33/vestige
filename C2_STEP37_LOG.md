# C2 Step 37 — Kalevala parallelism

Parent: `fd883f580d91cc7bb6e62ac5b4d8804d4e30f7a2` (Step 35 Mill agreement) on `perf/c2-step37-kalevala`, replayed from `d121d063` plus `steps7_10a`, `algos44`, and Steps 31–35. Hook order after the replay, verified in `retrieval.rs`: Step 31, Step 32, Step 33, Step 34 (`apply_c2_step34_bertillonage(ix, &q.text, &mut final_turn_indices, &t_scene)`), Step 35 (`apply_c2_step35_mill_agreement(ix, &plan, &mut final_turn_indices, &t_scene)`), then `maybe_dump_replay`. Step 37 is inserted after Step 35 and before `maybe_dump_replay`.

Datasets (hashes checked before any bench):

- LoCoMo `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- LongMemEval_S `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- BEAM-100K unpacked to `BEAM/chats/100K` (20 conversations)

## Dump fields (read in `write_one_replay_dump` before any ranker edit)

- `pre_ids` / `candidates`: seated turns first, then `t_scene` descending, `t_scene > 0`, cap 50. That is pool@50.
- `rank_pre` on a candidate is its index in that pool. For a seated turn it matches final rank, so it is not an independent pre-rank. `tools/c2_funnel.py` recomputes scene rank 0 as the argmax of `t_scene` (tie broken by turn index).
- `final_ids` is the seated list, the official turn order. Final rank 0 is `final_ids[0]`. Final top 5 is `final_ids[:5]`.
- `qtok_present` and `is_question` are dump fields. Ranking does not read `TurnIx.body`. Step 37 reads `TurnIx.content`, `form_digit`, `speaker`, session position, and content query cues (`query_cues` filtered by `is_content`).

## Stage funnel (LoCoMo, combined Step 31–35 tip, before the ranker edit)

Printed by `c2_funnel.py` on `retrieval_bench_tip35` (`5cc9d0f9e0e43247e64cf46c17e0a230b855686ab28418ff949c4d4b64ca3b69`):

| category | n | gold in corpus | gold in pool@50 | gold at scene rank 0 | gold at final rank 0 | gold in final top 5 |
|---|---:|---:|---:|---:|---:|---:|
| Multi-Hop | 282 | 282 | 243 | 72 | 97 | 158 |
| Open-Domain | 92 | 92 | 63 | 14 | 20 | 35 |
| Temporal | 321 | 320 | 299 | 182 | 191 | 248 |
| Single-Hop | 841 | 841 | 810 | 440 | 500 | 686 |

Step 37 does not change pool membership or `t_scene`. LoCoMo per-question hit bits are identical to the tip, so these five counts are also the winning binary's LoCoMo funnel. LME Turn@5 stays 428 and BEAM Turn@5 stays 238.

## Baseline (Steps 31+32+33+34+35 together, before Step 37)

First joint measurement of this stack. `retrieval_bench` release, `c2_bench.py`, 3 workers. Binary sha256 `5cc9d0f9e0e43247e64cf46c17e0a230b855686ab28418ff949c4d4b64ca3b69`.

| bench | metric | score |
|---|---|---|
| LME_S n=470 | Session any@1/5/10 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 |
| LME_S | Turn any@1/5/10 | 354/428/437 |
| LME_S | Turn All@5 | 314 |
| LoCoMo n=1536 | Turn any@1/5/10 | 808/1127/1226 |
| LoCoMo | Session any@1/5 | 1133/1416 |
| LoCoMo | Multi-Hop Turn@1 | 97/282 |
| LoCoMo | Open-Domain Turn@1 | 20/92 |
| BEAM-100K n=355 | Turn any@1/5/10 | 160/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 |
| BEAM-100K | Session any@1/5/10 | 133/249/275 |
| BEAM | multi_session_reasoning Turn@1 | 22/40 |

The published `bd2693e2` totals are LME Turn@1 350, LoCoMo Turn@1 796, BEAM Turn@1 156, with the same session totals. The joint tip is exactly those totals plus the listed flips and nothing else: LME `e8a79c70`, `c4a1ceb8`, `ccb36322`, `gpt4_2ba83207` (350+4=354); LoCoMo `5:13`, `2:106`, `3:143`, `9:147`, `6:25`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36` (796+12=808); BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0` (156+4=160). Every one of those questions is still Turn@1 = 1 on the joint tip. Session aggregates match `bd2693e2`. No composition repair.

LME Turn@1 by type at the tip: single-session-user 55/64, multi-session 90/121, single-session-preference 12/30, temporal-reasoning 93/127, knowledge-update 62/72, single-session-assistant 42/56.

## Attempt 1 — any direction, closed-class cues count (failed, not committed)

Approach: successive same-speaker turns with at most one turn between them, agreeing on `form_digit`, are one line when the shared content tokens outnumber half the shorter line. Slot 1 yields to the unique top-5 partner whose full `query_cues` (including closed class) strictly cover the head, whether that partner is earlier or later.

No commit SHA. Measured on release binary sha256 `698f6a77b831878d45ce64f344efc6393c506df1cb736e186cafcc2fadfa22bf`, then replaced.

Scoreboard vs the tip: LME Turn@1 354→354 (+1 −1). LoCoMo Turn@1 808→807. BEAM unchanged. Every Session, @5, @10, and All@5 figure is unchanged.

Gains (Turn@1 0→1): LME `gpt4_fa19884d`.

Losses (Turn@1 1→0): LME `gpt4_9a159967`; LoCoMo `0:13`.

Why it failed: on `gpt4_9a159967` the following declarative only added the closed-class cue `which` and displaced the gold question. On LoCoMo `0:13` the earlier Caroline line, which posts `career`, displaced the later decision that was already at slot 1. Numeral-shape agreement did not separate those pairs (both sides matched).

## Attempt 2 — following line, content cues only (WIN)

Same line test (successive same speaker, at most one intervening turn, shared content outnumbers half the shorter line, `form_digit` agrees, sets unequal). Slot 1 moves only when all of these hold:

- The partner is later in the session than the head. An earlier line does not take the seat.
- The partner is already in the top 5. Top-5 membership stays.
- Exactly one such partner qualifies.
- Its content query cues strictly cover the head's. A cue is a `query_cues` id whose cue-vocab word passes `is_content`. Closed-class tokens are not a reason to call one line richer.

`rsb_seat_behind` is not touched. No new ratio. The top-5 window is the existing `C2_BULLAE_WINDOW`. `DetHashSet` / the existing cue map only.

Files:

- `crates/vestige-core/src/algos_c2.rs` lines 3959–4052 (`apply_c2_step37_kalevala`, `step37_content_query_cues`, `step37_one_line`, `step37_short_restatement`) and lines 8002–8103 (unit test `step37_kalevala_seats_the_richer_parallel_line`)
- `crates/vestige-core/src/retrieval.rs` lines 6294–6298 (call after Step 35, before `maybe_dump_replay`)

Scoreboard vs the tip: LME Turn@1 354→355. Every other Turn, Session, All@5, and All@10 figure on all three benches is unchanged. LoCoMo and BEAM per-question vectors are identical to the tip (0 diffs). The only per-question change is LME `gpt4_fa19884d` Turn@1 0→1 (`[1,1,1,0,1,1,1]` → `[1,1,1,1,1,1,1]`). LME temporal-reasoning Turn@1 93→94. LME Session stays 430/463/468, so Session drops are 0. The Step 31–35 flips listed above stay hits.

Winning binary sha256 `1d7363188e0576173a5d6af36c4af5aa7db7b9863b019d40467979cd4e717b9f`.

What moved: slot 1 was `answer_ff201787_2#8` ("exploring more genres… recommendations for jazz"). The following user line `answer_ff201787_2#10`, already at rank 3, restates that opening and adds the content cue `started` ("started enjoying their music today"). That line is the gold turn. They agree on numeral shape (neither has a digit). One intervening assistant turn sits between them.

Gates on this commit:

- `cargo clippy -p vestige-core --all-targets -- -D warnings` clean
- `cargo test -p vestige-core`: 565 passed, 0 failed
- `cargo check -p vestige-mcp` passed
- `c2_hashcheck.py`: ranked ids 0/500, full responses 0/500
