# C2 Step 35 — Mill's method of agreement

Parent: `5ffa0fdd2eb9cd51f1e4e85de56cd7a1d6ba7766` (Step 33 Triangulation) on `perf/c2-step35-mill-agreement`, replayed from `d121d063` plus `steps7_10a`, `algos44`, and Steps 31–33. Hook order in `retrieval.rs` after the replay: Step 31, Step 32, Step 33, Step 35, `maybe_dump_replay`.

Datasets (hashes checked before any bench):

- LoCoMo `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- LongMemEval_S `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- BEAM-100K unpacked to `BEAM/chats/100K` (20 conversations)

## Dump fields (read in `write_one_replay_dump` before any ranker edit)

- `pre_ids` / `candidates`: seated turns first, then `t_scene` descending, `t_scene > 0`, cap 50. That is pool@50.
- `rank_pre` on a candidate is its index in that pool. For a seated turn it matches final rank, so it is not an independent pre-rank. `tools/c2_funnel.py` recomputes scene rank 0 as the argmax of `t_scene` (tie broken by turn index).
- `final_ids` is the seated list, the official turn order. Final rank 0 is `final_ids[0]`. Final top 5 is `final_ids[:5]`.
- `qtok_present` is dump-only. Ranking does not read it and does not read `TurnIx.body`. Step 35 attests a surface anchor only when the anchor's vocab id is in the turn's `content` or `feats`.

## Stage funnel (LoCoMo, combined Step 31+32+33 tip, before the ranker edit)

Printed by `tools/c2_funnel.py` on the tip binary:

| category | n | gold in corpus | gold in pool@50 | gold at scene rank 0 | gold at final rank 0 | gold in final top 5 |
|---|---:|---:|---:|---:|---:|---:|
| Multi-Hop | 282 | 282 | 243 | 72 | 97 | 158 |
| Open-Domain | 92 | 92 | 63 | 14 | 20 | 35 |
| Temporal | 321 | 320 | 299 | 182 | 189 | 248 |
| Single-Hop | 841 | 841 | 810 | 440 | 498 | 686 |

The winning rule changes no LoCoMo question, so this funnel is also the winning binary's LoCoMo funnel. LME Turn@5 stays 428 and BEAM Turn@5 stays 238; only LME Turn@1 moves, by one question.

## Baseline (Step 31+32+33 together, before Step 35)

Measured with `retrieval_bench` release, `c2_bench.py`, 4 workers. The listed flips vs `bd2693e2` are all still hits (Turn@1 = 1). Net movement vs the published `bd2693e2` totals is exactly those flips: LME Turn@1 350→352, LoCoMo Turn@1 796→804, BEAM Turn@1 156→159. Session totals match `bd2693e2`. No composition repair was required.

| bench | metric | score |
|---|---|---|
| LME_S n=470 | Session any@1/5/10 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 |
| LME_S | Turn any@1/5/10 | 352/428/437 |
| LME_S | Turn All@5 | 314 |
| LoCoMo n=1536 | Turn any@1/5/10 | 804/1127/1226 |
| LoCoMo | Session any@1/5 | 1133/1416 |
| LoCoMo | Multi-Hop Turn@1 | 97/282 |
| LoCoMo | Open-Domain Turn@1 | 20/92 |
| BEAM-100K n=355 | Turn any@1/5/10 | 159/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 |
| BEAM-100K | Session any@1/5/10 | 133/249/275 |
| BEAM | multi_session_reasoning Turn@1 | 21/40 |

LME Turn@1 by type at the tip: single-session-user 54/64, multi-session 89/121, single-session-preference 12/30, temporal-reasoning 93/127, knowledge-update 62/72, single-session-assistant 42/56.

## Attempt 1 — literal cover (failed, not committed)

Approach: for each surface content-group label (not a `\u{1}` stem, hypernym, or entity phrase), the hit set is pool turns whose `content` or `feats` contain that vocab id. Exactly one session in every hit set is the agreement. If the head is outside it, the unique top-5 turn of that session that posts every anchor the head posts takes slot 1. Ties and several common sessions do nothing.

No commit SHA. The rule was measured on a release binary and then replaced in the working tree.

Scoreboard vs the tip: LME Turn@1 352→351, LoCoMo Turn@1 804→800, BEAM Turn@1 159→155. Session metrics and every @5/@10 were unchanged.

Gains (Turn@1 0→1): LME `3a704032`, LME `gpt4_2ba83207`.

Losses (Turn@1 1→0): LME `36b9f61e`, `60036106`, `a9f6b44c`; LoCoMo `2:92`, `8:76`, `3:160`, `5:118`; BEAM `18:knowledge_update:1`, `20:temporal_reasoning:1`, `16:preference_following:1`, `7:multi_session_reasoning:1`.

Why it failed: a head that already posted some anchors (often the gold turn) was displaced by another session's turn that merely covered those anchors and added one more, including an other-role echo. `20:temporal_reasoning:1`, a Step 32 flip, was among the losses. Covering the head is not agreement about the phenomenon.

## Attempt 2 — non-instance head, or complete same-speaker instance (WIN)

Same agreement (surface anchors, pool of 50, exactly one common session, head outside it, top-5 membership unchanged). Slot 1 moves in one of two shapes:

- Non-instance. The head posts no surface anchor, so it is not an instance. The unique top-5 turn of the agreement session that posts at least one anchor takes slot 1.
- Complete instance. The head posts some anchors and misses others. The unique top-5 turn of the agreement session that posts every anchor, and whose speaker field equals the head's, takes slot 1. A different speaker is the other role's echo, not a second witness.

Two qualifiers are a tie. A head inside the agreement session stays. `rsb_seat_behind` is not touched. No new ratio. `DetHashSet::default()` only.

Files:

- `crates/vestige-core/src/algos_c2.rs` lines 3580–3725 (`c2_surface_anchor_ids`, `c2_dump_pool`, `apply_c2_step35_mill_agreement`) and lines 7330–7529 (unit test `step35_mill_agreement_seats_the_unique_common_session`)
- `crates/vestige-core/src/retrieval.rs` lines 6282–6288 (call after Step 33, before `maybe_dump_replay`)

Scoreboard vs the tip: LME Turn@1 352→353. Every other Turn, Session, All@5, and All@10 figure on all three benches is unchanged. LoCoMo and BEAM per-question vectors are identical to the tip (0 diffs). The only per-question change is LME `gpt4_2ba83207` Turn@1 0→1 (`[1,1,1,0,1,1,0]` → `[1,1,1,1,1,1,0]`). LME multi-session Turn@1 89→90. LME Session stays 430/463/468, so Session drops are 0. The Step 31–33 flips listed above stay hits.

Question: `gpt4_2ba83207`, "Which grocery store did I spend the most money at in the past month?" The head posted no surface anchor. `spend` is witnessed in one pool session, and that session is the intersection of `grocery`, `store`, `spend`, and `money`. Its unique top-5 instance is gold turn `answer_6a3b5c13_4#6`.

Gates on this tree:

- `cargo clippy -p vestige-core --all-targets -- -D warnings` exit 0
- `cargo check -p vestige-mcp` exit 0
- `cargo test -p vestige-core` exit 0 (563 lib tests, 0 bin tests, 3 integration tests, doctests)
- `c2_hashcheck.py` ranked ids 0/500 and full-response hashes 0/500

Winning commit SHA is the commit that adds this file. See `git log -1`.
