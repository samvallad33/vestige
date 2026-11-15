# C2 Step 31 — Exchequer split tally

Branch `perf/c2-step31-exchequer`. Base of the applied series is `d121d063be3f22c59e40b6f39d7a14c905569a43`. `git am -3` applied `steps7_10a` (5) then `algos44` (44) with no conflicts. Author on every applied commit is Sam Valladares `<samvalladares21@gmail.com>`. The am tip is `e495b88109adfbd8835a80ae010dd98c8ab158bf` (tree `4612c06f74b115c06606306e9e84abeacb6de956`). Object `bd2693e2ad98d7a1c87973e239e7935b9d9596c7` is not in this clone; the last patch's `From` line is that commit, and the series applied cleanly onto the stated base. The scoreboard below matches Computer 2's bd2693e2 numbers on every published metric.

Datasets, hashes checked before any bench:

- `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM-100K: 21 conversations under `BEAM/chats/100K`

## Dump fields (verified in `write_one_replay_dump`, retrieval.rs)

`VESTIGE_REPLAY_DUMP` is off the ranking path. `maybe_dump_replay` snapshots the seated list and `t_scene` after every post-gatekeeper seat. `write_one_replay_dump` then builds:

- `final_ids` / `seated_ids`: official turn order after the gatekeeper and every C2 seat. `rank` is the index in that list.
- `pre_ids`: that seated list first, then `t_scene`-descending turns with `t_scene > 0` until 50. `rank_pre` is the index in `pre_ids`. So `pre_ids[0]` is `final_ids[0]` by construction, and `pre_ids[:k]` equals `final_ids` whenever the seated list is at least `k` long. A literal "gold at pre_rank 0" is the same count as "gold at final rank 0".
- `t_scene`: scene-pooled turn score copied onto the snapshot. It is not rewritten by seating. Scene-rank 0 is `argmax t_scene` (tie break `idx`), which is the pre-seating order. The shipped `c2_funnel.py` prints this as "gold at scene rank0" and says so in its header.
- There is no post-Kemeny snapshot. `rank1_gatekeeper_keep_rsb_head` Kemeny-orders slots `1..k` with slot 0 frozen, then #19–#44 and Step 31 run. The dump sees only the final seated list.
- `parent_user_turn_id`: `TurnIx.answers` (previous different-speaker turn). `response_turn_id`: the next turn whose `answers` edge points here. Both are structural, not body reads.
- `qtok_present` and `turn_ic_novel` read `TurnIx.body` and are dump-only. Ranking does not.

Baseline literal funnel from the bd2693e2 dumps (gold id in the field):

| bench | n | pool@50 | pre_rank@1 (= final@1) | pre_ids top5 | final@5 | scene-rank@1 | scene top5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LME_S | 470 | 463 | 350 | 428 | 428 | 265 | 399 |
| BEAM-100K | 355 | 328 | 156 | 239 | 238 | 124 | 221 |

BEAM `pre_ids` top5 is 239 and final@5 is 238 because one seated list is shorter than 5, so the pool fill occupies a `pre_ids` slot the official list does not have. LME pool@50 463 and BEAM pool@50 328 match `missmap-bd2693e2.txt`.

LoCoMo 5-line funnel from `tools/c2_funnel.py` on the baseline binary, before any ranker edit (scene-rank 0 is argmax `t_scene`, not `rank_pre`):

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
  gold at final rank0 187
  gold in final top5 248
[LoCoMo Open-Domain] n=92
  gold in corpus     92
  gold in pool@50    63
  gold at scene rank0 14
  gold at final rank0 18
  gold in final top5 35
[LoCoMo Single-Hop] n=841
  gold in corpus     841
  gold in pool@50    810
  gold at scene rank0 440
  gold at final rank0 494
  gold in final top5 686
```

LoCoMo speakers are only the 18 names in `locomo10.json` (Caroline, Melanie, …). No turn has role `user` or `assistant`, so a seat that requires `speaker == "assistant"` cannot fire there.

## Baseline scoreboard (binary of am tip `e495b881`, before Step 31)

Reproduced exactly.

- LME_S n=470: Session 430/463/468, All@5 424, All@10 447. Turn 350/428/437, All@5 314.
- LoCoMo n=1536: Turn 796/1127/1226, Session 1133/1416.
- BEAM-100K n=355: Turn 156/238/267, All@5 118, All@10 138, Session 133/249/275.

## What the misses actually are

LME `single-session-assistant` is 56 questions, Turn@1 41, so 15 misses (11 in top 5, 2 in k not top 5, 2 outside k). On the top-5 misses the usual head is a later assistant turn that repeats the query's content groups in one message. The gold assistant often states the asked fact (the egg count, the bottle list) while the user turn it answers holds the rest of the query. BEAM `instruction_following` gold ids are user turns (34 misses: 12 in top 5). Those questions are mostly not `assistant_attribution`, and seating an assistant turn cannot put a user gold at slot 1. LoCoMo has no assistant role.

## Simulations that were not shipped

All of these were scored on the baseline replay dumps (content-token overlap as a proxy for `c2_turn_groups`). Each one moved at least one current Turn@1 hit to a miss, so none was committed.

- Reply-of-head: if slot 1 is the user and its assistant reply is a both-halves turn in the top 5, promote the reply. Without the attribution gate, LME +1/−141. With attribution and not preference, LME +1 (`1903aded`) / −1 (`5809eb10`, the gold is the user turn).
- Any both-halves assistant over a one-sided head, attribution only. LME +1/−5.
- Highest geometric mean among both-halves assistants, even when the head is already both-halves. LME +2 (`1903aded`, `3249768e`) / −6. The extra gain is real and the losses are real; it is a different rule, not a threshold tweak, and it fails the zero-loss gate.
- Split (each half has a group the other lacks) whose union covers the head, without the preference cut. LME +1 (`e8a79c70`) / −1 (`35a27287`, "Can you recommend…", `preference=true`).

## Attempt 1 — shipped rule — WIN

`apply_c2_step31_exchequer_split` in `crates/vestige-core/src/algos_c2.rs` lines 2398–2495. Hook in `crates/vestige-core/src/retrieval.rs` lines 6255–6259, inside the existing `assistant_attribution && !is_event_sequence` block, after `apply_c2_attribution_unseated_opener_t10_entry`. Unit test `exchequer_split_seats_complementary_assistant_over_echo` at lines 6496–6576. `rsb_seat_behind` is not called and is not edited. `t_scene` is read only.

Rule. On `assistant_attribution`, and not preference, temporal, summary, WH-person, event-sequence, or a named-speaker session: for each assistant in slots 1..4 whose `answers` parent is a user turn already in the official list, foil = that assistant's query groups (`c2_turn_groups` / `c2_group_ic`, index token stats, never `TurnIx.body`), stock = the parent user's groups. A split is a pair where each half attests a group the other lacks. The tally is `sqrt(coverage(foil) * coverage(stock))`. The split with the highest tally whose union covers every group the incumbent attests takes slot 1. Tie break: higher `t_scene`, then earlier turn index. Top-5 membership is unchanged, so Turn@5, Turn@10, All@5, and the session list cannot move.

Real run of `/workspace/c2-bins/step31a` (sha256 `d01b44eb1611b3bb887d3e0f27faaf831ab95c5c8f36f3e9e65a7602ddb64373`), `python3 tools/c2_bench.py --workers 4`:

- LME_S Session 430/463/468, All@5 424, All@10 447. Turn **351**/428/437, All@5 314. single-session-assistant Turn@1 41 → 42.
- LoCoMo Turn 796/1127/1226, Session 1133/1416.
- BEAM-100K Turn 156/238/267, All@5 118, All@10 138, Session 133/249/275.

Per-question bits vs the baseline json: LME Turn@1 +1 −0 (`e8a79c70` only). Every other LME bit, all 1536 LoCoMo bits, and all 355 BEAM bits: +0 −0. LME Session drops: 0.

`e8a79c70` ("how many eggs did you say we need for the classic French omelette"): gold `answer_ultrachat_13075#1` is the assistant ingredient list ("2-3 eggs"). Its stock is user `#0` ("classic French omelette"). The old head `#5` is a later fillings reply that restates classic/french/omelette/eggs in one turn. Gold was rank 4; it is now rank 0. Turn@5 was already 1 and stayed 1.

Post-change stage picture. The seat never changes who is in the top 5 or the pool, and it never writes `t_scene`, so pool@50, final@5, and scene-rank are the baseline numbers above. final@1 / pre_rank@1 on LME is 351. LoCoMo and BEAM final@1 are unchanged (796 and 156).

Gates on this tree:

- `cargo clippy -p vestige-core --all-targets -- -D warnings` exit 0
- `cargo check -p vestige-mcp` exit 0
- `cargo test -p vestige-core` 558 passed, 0 failed
- `tools/c2_hashcheck.py` ranked ids 0/500, full response including f32 scores 0/500

Winning commit is the one commit on top of `e495b88109adfbd8835a80ae010dd98c8ab158bf` that adds `apply_c2_step31_exchequer_split`. Its hash is `git rev-parse HEAD` of the commit that contains this file.
