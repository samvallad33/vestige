# C2 pool A log

Focus: LoCoMo pool admission and top-5 seating. Baseline is the combined Step 31+32+33+34 tip, measured before any new seat. Author of the seat commit: Sam Valladares.

## Baseline (measured, not assumed)

Hook order in `rank_one`: Step 32 trial balance, Step 33 triangulation, Step 34 Bertillonage, then `maybe_dump_replay` with `t_own`, `t_turn_raw`, `t_scene_sess`, `s_score`, `bridge_boost`.

| Bench | Score |
| --- | --- |
| LME_S | Session 430/463/468 All@5 424 All@10 447; Turn 353/428/437 All@5 314 |
| LoCoMo | Turn 808/1127/1226; Session 1133/1416 |
| BEAM-100K | Turn 160/238/267 All@5 118 All@10 138; Session 133/249/275 |

Against the algo-44 tree (published bd2693e2 board: LME Turn@1 350, LoCoMo 796, BEAM 156), the tip has 0 per-question coordinate losses (LME +3, LoCoMo +12, BEAM +4 coordinate gains). Listed flips still Turn@1: LME `e8a79c70`, `c4a1ceb8`, `ccb36322`; LoCoMo `5:13`, `2:106`, `3:143`, `9:147`, `6:25`, `4:28`, `4:44`, `8:124`, `1:42`, `5:105`, `9:9`, `9:36`; BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1`, `19:multi_session_reasoning:0`.

Stage funnel before the seat (gold in corpus / pool@50 / scene-rank 0 / final rank 0 / final top 5):

- Multi-Hop 282 / 243 / 72 / 97 / 158
- Temporal 321 / 320 corpus, pool 299 / 182 / 191 / 248
- Open-Domain 92 / 63 / 14 / 20 / 35
- Single-Hop 841 / 810 / 440 / 500 / 686

Census: `POOL_MISS_CENSUS_A.md`. Indexed golds missing pool@50 are below the `t_scene` cut (533 turns). Posting drops: 0. Session-candidate-set drops: 0. Four evidence ids are not in the index.

## Attempt 0 — simulated seats, not committed

Offline on the tip dumps. A fire that changes a hit bit is a gain or a loss. Rules with any loss were not implemented.

| Rule | Fires | Gains | Losses | Why it stopped |
| --- | ---: | --- | --- | --- |
| Promote max-`t_own` from ranks 5–9 into the min-`t_own` slot of 1–4 | 99 | T@5 +5 | T@5 −2 | Losses |
| Same, but only a non-question over a question | 937 | T@5 +22 | T@5 −17 | Losses |
| Admit max-`t_own` outside the official list into the tail, 2× the tail | 115 | T@10 +8 | T@10 −1 | One loss |
| Next-turn of a top-5 question replaces a weaker top-5 question | 174 | T@5 +4, T@10 +3 | T@5 −3, T@10 −4 | Losses |
| Unique max-`t_own` inside top 10 takes slot 0 when slot 0 is a question | 54 | T@1 +7 | T@1 −9, T@5 −3 | Losses |
| Session-rank-0 turn inside top 5 swaps with slot 0 | — | T@1 +15 | T@1 −38 | Losses |

## Attempt 1 — buried lexical peak — WIN

A declarative in a named-speaker session whose `t_own` strictly exceeds every turn in the scene pool of 50, and which is the only such turn, takes the lowest-`t_own` question seat in slots 1–4. Slot 0 stays. The displaced question is kept (appended under `k`, otherwise written over the tail). The session list is not rewritten. User/assistant transcripts never fire (`c2_named_session`).

Offline simulation on the tip: 11 fires, T@5 +1, T@10 +1, losses 0. The release bench matches that delta.

- Files: `crates/vestige-core/src/algos_c2.rs` (`apply_c2_buried_lexical_peak`, about 3809–3896, test `buried_lexical_peak_seats_the_unique_unpooled_declarative`), `crates/vestige-core/src/retrieval.rs` (call after Step 34, about 6287–6298, plus dump-only `score_rows` / `session_rows`).
- Funnel before the seat: see above. After, measured with `tools/c2_funnel.py` on the release binary: Multi-Hop 282 / 243 / 72 / 97 / 158, Temporal 320 corpus / pool 299 / 182 / 191 / 248, Open-Domain 92 / 63 / 14 / 20 / 35, Single-Hop 841 / 811 / 440 / 500 / 687. Only Single-Hop pool@50 (810→811) and final top 5 (686→687) move.
- Scoreboard: `SCOREBOARD_POOL_A.md`. Per-question diff: `POOL_A_PERQ_DIFF.md`.
- Gains: LoCoMo `9:137` T@5 and T@10. Losses: none.
- Gates: clippy `-D warnings` clean, `cargo check -p vestige-mcp` clean, `cargo test -p vestige-core` 564 passed, LME Session drops 0, hash-check below.

### Hash-check

`tools/c2_hashcheck.py` on the release binary, two worker counts:

- hash-check (ranked ids): 0/500 differing questions
- hash-check (full response incl. f32 scores): 0/500 differing questions

## Attempt 2 — anaphoric tail — WIN

The strongest non-question in ranks 5..9 of a named-speaker session, when that turn is anaphoric and its own score strictly beats every top-5 own score, swaps with the lowest-own question in slots 1..4. The swap stays inside the returned list. Slot 0 and the session list stay. Anaphora is the existing ingest flag (a closed-class continuation or pronoun in the first tokens). The seat does not read turn text.

Offline on the post-peak dumps, the unguarded tail-over-question swap fired 83 times: Turn@5 +4 / −2. Requiring the tail maximum to be anaphoric kept all 4 gains and dropped both losses (68 fires). Other simulated seats were not this commit: a question's immediate reply (+5 / −9 Turn@5), a void-session representative (0 fires), an unseated pool declarative over a redundant question (+14 / −14), and an outside-list peak with no session guard (+4 / −2 Turn@5). Two further 0-loss structures were re-simulated on top of this seat and left for the next commit: digit-shape agreement (Temporal `4:24`, Turn@5 +1 and Turn@10 +1) and an unseated session peak whose session is absent from the top 5 (Temporal `1:10` and `6:64`, Turn@5 +2 and Turn@10 +2).

- Files: `crates/vestige-core/src/algos_c2.rs` (`apply_c2_anaphoric_tail`, 3905–3969, test `anaphoric_tail_swaps_the_dominant_pronoun_with_the_weakest_question`), `crates/vestige-core/src/retrieval.rs` (call after the buried peak, 6299–6303).
- Release bench versus the buried-peak commit: LoCoMo Turn@5 1128→1132. Turn@1, Turn@10, and both session metrics stay. LME and BEAM coordinate changes: 0. LME session-coordinate drops: 0.
- Funnel after this seat, measured: Multi-Hop 282 / 243 / 72 / 97 / 158, Temporal 320 corpus / pool 299 / 182 / 191 / 248, Open-Domain 92 / 63 / 14 / 20 / 35, Single-Hop 841 / 811 / 440 / 500 / 691. Only Single-Hop final top 5 moves (687→691).
- Versus the Step 31+32+33+34 tip: LoCoMo Turn 808/1132/1227, Session 1133/1416. Single-Hop Turn@5 686→691. Losses: 0 on every coordinate of all three benches.
- Gains versus the buried peak: LoCoMo `7:146`, `9:113`, `9:131`, `9:146`, each Turn@5 only. All four are Single-Hop and were already Turn@10 hits.
- Gates: clippy `-D warnings` clean, `cargo check -p vestige-mcp` clean, `cargo test -p vestige-core` 565 passed, hash-check 0/500 ranked ids and 0/500 full responses.

### Hash-check

`tools/c2_hashcheck.py` on this release binary:

- hash-check (ranked ids): 0/500 differing questions
- hash-check (full response incl. f32 scores): 0/500 differing questions
