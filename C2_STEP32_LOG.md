# C2 Step 32 — Trial balance

One commit on top of algos #44 (`c06dbfc3770d17dee228291adcd1c136ce8f5cb7`).
Author Sam Valladares. Failed measurements below were not committed.
The winning rule is `apply_c2_step32_trial_balance`
(`crates/vestige-core/src/algos_c2.rs`, the function and its three tests).
The hook is the call immediately before `maybe_dump_replay` in
`crates/vestige-core/src/retrieval.rs`.

## Stage funnel (bd2693e2 tree, before any Step 32 edit)

Dump field check, from `write_one_replay_dump`:

- `pre_ids` / `candidates` are the seated final order, then `t_scene` fill out to 50.
- `pre_ids[0] == final_ids[0]` on 2361/2361 records, so dump rank 0 is final rank 0.
- `final_ids` is post-Kemeny and post every C2 seat. There is no separate post-Kemeny snapshot.
- `qtok_present` is dump-only (body token or content id). Ranking does not read it and does not read `TurnIx.body`.
- Scene rank is argmax `t_scene` (`order()`: score desc, session ts desc, index asc). It is not the seated order.

| bench | n | gold in corpus | pool@50 | pre_rank 0 = final 0 | final top5 | scene rank 0 | scene top5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LME_S | 470 | 470 | 463 | 350 | 428 | 265 | 399 |
| LoCoMo | 1536 | 1535 | 1415 | 796 | 1127 | 708 | 1110 |
| BEAM-100K | 355 | 355 | 328 | 156 | 238 | 124 | 218 |

BEAM `pre_ids[:5]` was 239 against final top5 238: a short seated list is scene-filled. LME and LoCoMo `pre_ids[:5]` matched final top5 (428, 1127).

Baseline scores matched Computer 2 exactly before the ranker edit. See `SCOREBOARD_STEP32.md`.

## Rule

Accounts are the query's lexical content groups (anchor tokens). A group is posted only by its surface lexeme — the group label — in the turn's content or features. A stem, a hypernym, and a typo fallback do not post it. Entity phrases (`\u{1}ent:`) are not a second account. `TurnIx.body` is never read. Speaker name is not an input.

Opener arm. Rarest surface lexeme = highest turn-own idf; the lowest group index wins an exact tie. Slot 1 is unbalanced when it is a question that does not post that lexeme. A top-5 session opener whose session is not later takes slot 1 when it posts the rare lexeme and every surface lexeme slot 1 already posts. Several openers: highest `t_scene`, then earliest session, then lower turn index.

Echo arm, only when the opener arm does not move. Slot 1 is the direct reply (`answers`, and the immediately previous turn of the same session) to a declarative turn already in the top 5. That turn, together with this reply, posts every surface lexeme, and slot 1 together with its own reply does not. The answered turn itself posts every surface lexeme rarer than the median surface idf, and slot 1 does not. That turn takes slot 1.

Top-5 membership does not change. The session list is not rewritten. `rsb_seat_behind` is not touched. Hook: `retrieval.rs` immediately before `maybe_dump_replay`, after every other C2 seat.

## Attempts

### A — every content group, in the turn or its uptake

Credit self, uptake, or the pair. Accounts: all content tokens, idf ≥ 1.5, idf ≥ 2.2, above-median, entity, entity∪rarest, rarest. Pick: highest scene or earliest session. Every variant had more losses than gains. The least-bad raw cell was rarest / self / scene at +44 / −246. Requiring every query word promotes the turn that echoes the question. Gold answers omit the queried name and the framing word. Not committed.

### B — 6912 structural cuts of the same coverage test

Scene floor, speaker-name tokens dropped, uptake-only, same session, other session, question head, declarative candidate. No zero-loss configuration had a gain. Closest: all tokens / pair credit / uptake / scene floor, LoCoMo +2 / −1. Not committed.

### C — orthographic particulars

A later same-session turn that posts a new mid-sentence capital or a digit run. +15 / −157. Restricted to a head with no particular: +23 / −54. Not committed.

### D — group credit, including stems and hypernyms

Same opener shape as the winner, but `c2_turn_groups` lets a stem or a hypernym post the account. Real bench: BEAM Turn@1 156 → 159, zero losses, LME and LoCoMo unchanged. "recipe" posted the "recipes" account, and the LME gain did not fire. Not the shipped rule.

### E — surface lexeme, opener arm only

Real bench, zero per-question losses:

- LME Turn@1 350 → 351 (`c4a1ceb8`)
- LoCoMo Turn@1 796 → 797 (`5:13`)
- BEAM Turn@1 156 → 159 (the three questions in the diff)

Every other metric flat. Superseded in the same commit by the echo arm, which keeps these five.

### F — winner: surface opener, then echo

Real bench against the bd2693e2 baseline. Zero per-question losses on every recorded metric (LME S@1/5/10, T@1/5/10, Turn All@5; LoCoMo S@1/5, T@1/5/10; BEAM T@1/5/10, All@5, S@1/5, Session All@5). Aggregate deltas are only Turn@1:

- LME 350 → 351
- LoCoMo 796 → 801
- BEAM 156 → 159

Nine questions flipped to a hit. None flipped to a miss. The echo arm added LoCoMo `2:106`, `3:143`, `9:147`, and `6:25` on top of attempt E. `6:25` was a gain the dump-token simulation did not predict; the surface-lexeme books caught it and it did not cost another question.

## Gates

- `cargo clippy -p vestige-core --all-targets -- -D warnings` clean
- `cargo check -p vestige-mcp` clean
- `cargo test -p vestige-core` 560 passed, 0 failed
- hash-check, two independent worker passes over all 500 LME_S questions: 0/500 differing ranked ids, 0/500 differing full responses
- LME Session stayed 430/463/468, Session All@5 stayed 424
