# Retrieval sanitizer receipts

Working branch: `cursor/skeptic-ranker-eb3b`.
Start commit: `57acdc468a352e9be12a7ae6e1e10901902798ac` (PR 753 head, parent `8bce7116`).
Commit author for this branch: Sam Valladares <samvalladares21@gmail.com>.

No `p2_mask`, `p2_on`, `VESTIGE_P2_MASK`, `apply_move2_qty_slot_split`, `apply_move3_heideltime_anchor`, or `dl1_zero_df_vocab_id` exists at this start point. Searched the tree. Nothing to delete for R1.

No hardcoded phrases `before i got`, `am i planning`, `have i had`, `did i get`, or `record in` exist in the ranker. Nothing to delete for R2.

## What the dump fields mean

`VESTIGE_REPLAY_DUMP` does not change ranking. It is read only after scores exist.

`rank_pre` on a candidate is the index of that turn in `candidates[]`. That array is built in `write_one_replay_dump` by copying the **final** seated list and then appending other positive-`t_scene` turns until length 50. For every turn that was returned, `rank_pre` equals its final rank. It is not a pre-rank stage and it is not a pre-Kemeny stage.

The harness sends `top_k: 10`. Metrics at @10 cannot see anything the worker did not return. Some questions return 11 turns: ReplySource can insert the prompting user turn without shrinking the list back to `k`. @10 still reads only the first 10 ids.

## Phase 0 — baseline, no ranker edit

Binary built from `57acdc46` before any edit on this branch.

- SHA: `57acdc468a352e9be12a7ae6e1e10901902798ac`
- binary sha256: `fa9cf252c871008b17dd2011e590ad6598442cc60b6bba5aee769d2edd044341`
- paths: `benchmarks/results/sanitize-baseline/board-seed1.json`, `board-seed2.json`, `rankings-seed1.jsonl`, `rankings-seed2.jsonl`, `phase0.log`, `retrieval_bench.sha256`
- symbols: none. `crates/vestige-core/src/retrieval.rs` was not in this commit.

### Input check (seed 1)

`top_k` sent by the scorer is 10. "Gold in returned list" is gold inside that returned window. It is not pool@50.

| slice | n | turns len min/mean/max | gold in corpus | gold in returned | gold absent |
|---|---:|---|---:|---:|---:|
| LME session, non-abstention | 470 | 10 / 10.03 / 11 | 470 | 467 | 3 |
| LME turn (`has_answer`), non-abstention | 470 | 10 / 10.03 / 11 | 470 | 417 | 53 |
| LoCoMo tune turns | 997 | 10 / 10 / 10 | 996 | 816 | 181 |
| BEAM turns | 355 | 1 / 10.13 / 11 | 355 | 254 | 101 |

One LoCoMo tune gold turn id is not in the indexed turns (evidence id with no ingested `dia_id`/`text`).

### Board (identical on `RUST_HASH_SEED=1` and `RUST_HASH_SEED=2`)

LME_S, 470 non-abstention, 30 `*_abs` excluded. Counts are hits, then percent.

| metric | hits | percent |
|---|---:|---:|
| Session Any@1 | 428 | 91.06 |
| Session Any@5 | 463 | 98.51 |
| Session Any@10 | 467 | 99.36 |
| Session All@5 | 420 | 89.36 |
| Turn Any@1 | 300 | 63.83 |
| Turn Any@5 | 396 | 84.26 |
| Turn Any@10 | 417 | 88.72 |

Turn gold is `has_answer` on `longmemeval_oracle.json`, aligned onto `longmemeval_s_cleaned.json` by question id and session id. Content of those turns matched. The cleaned file's turns do not carry `has_answer`; the scorer reads it only after ranking.

LoCoMo tune is dialogues 1–5 (`conv-26,30,41,42,43`), 997 scored, 2 without evidence excluded. Overall is all 10 dialogues, 1982 scored, 4 without evidence excluded. **Overall includes the test split (dialogues 6–10).** Those aggregates were printed once for this baseline. They are not a tuning signal. The test split is not scored again until the frozen binary.

| slice | n | Sess R@1 | Sess R@5 | Exact R@1 | Exact R@5 | Exact R@10 | Soft R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tune all cats | 997 | 77.83 | 93.98 | 51.15 | 75.63 | 81.85 | 81.14 |
| tune core (not cat 5) | 760 | 75.26 | 93.16 | 50.79 | 72.89 | 79.87 | 78.68 |
| tune Multi-Hop | 142 | 50.70 | 88.73 | 32.39 | 54.23 | 64.08 | 62.68 |
| tune Temporal | 156 | 82.05 | 97.44 | 69.23 | 84.62 | 89.10 | 86.54 |
| tune Open-Domain | 44 | 45.45 | 68.18 | 22.73 | 38.64 | 45.45 | 45.45 |
| tune Single-Hop | 418 | 84.21 | 95.69 | 53.11 | 78.47 | 85.41 | 84.69 |
| tune Adversarial | 237 | 86.08 | 96.62 | 52.32 | 84.39 | 88.19 | 89.03 |
| overall all cats | 1982 | 75.93 | 93.09 | 48.64 | 73.81 | 80.52 | 79.52 |
| overall Multi-Hop | 282 | 54.96 | 86.52 | 30.50 | 51.42 | 63.83 | 58.87 |
| overall Open-Domain | 92 | 38.04 | 70.65 | 20.65 | 36.96 | 43.48 | 43.48 |
| overall Temporal | 321 | 74.14 | 92.52 | 56.07 | 73.52 | 78.82 | 79.13 |
| overall Single-Hop | 841 | 83.59 | 96.31 | 53.86 | 80.14 | 86.92 | 85.97 |
| overall Adversarial | 446 | 83.86 | 96.19 | 50.67 | 83.86 | 87.89 | 88.12 |

BEAM 100K: 20 conversations, 400 questions, 355 scored (empty `source_chat_ids` dropped), 2127 sub-sessions, 5732 turns.

| metric | hits | percent |
|---|---:|---:|
| Session Any@1 | 128 | 36.06 |
| Session Any@5 | 247 | 69.58 |
| Session Any@10 | 276 | 77.75 |
| Session All@5 | 143 | 40.28 |
| Turn Any@1 | 148 | 41.69 |
| Turn Any@5 | 222 | 62.54 |
| Turn Any@10 | 254 | 71.55 |

### Determinism

`RUST_HASH_SEED=1` vs `RUST_HASH_SEED=2`, 2841 ranked lists (500 LME + 1986 LoCoMo + 355 BEAM).

- 2839 lists byte-identical
- 2 BEAM turn lists differ; session lists for those two questions are identical; headline rates did not move
  - `beam 1#9` rank 10: turn `31` vs `143`
  - `beam 2#10` ranks 6–8: `111,44,79` vs `44,79,111`

LME and LoCoMo were fully identical across the two seeds.

### Funnel

Not measured on the pristine binary. `rank_pre` is not pre-rank. The next commit adds stage ids and prints the funnel. That commit must not move this board.

### Hash seed does not pin `HashMap`

On rustc 1.98.1, `HashMap::new()` draws a fresh thread-local seed on every map. Two processes with `RUST_HASH_SEED=1` produced different key orders. A dump-off rerun of the instrumented binary differed from seed 1 on one list only: BEAM `2#10` ranks 6–8, the same triple that already differed between seed 1 and seed 2. Headline rates did not move. `RUST_HASH_SEED` is not a determinism proof on this toolchain. E1 has to remove the iteration dependency.

## Stage snapshots (measurement only)

Commit `7343012036d5ad337e07d5d69215f852e1c8c393`.

Symbols in `crates/vestige-core/src/retrieval.rs`:

- `rank_one`: `session_pool50` (line 5407, `s_score + bridge_boost`, top 50, score > 0), `turn_pool50` (line 5565, `t_scene`, top 50, score > 0), `pre_kemeny` (line 5948, clone of `final_turn_indices` immediately before `rank1_gatekeeper_keep_rsb_head`)
- `maybe_dump_replay` (line 6787), `write_one_replay_dump` (line 6945) writes `stage_turn_pool50_ids`, `stage_turn_pre_kemeny_ids`, `stage_turn_post_kemeny_ids`, `stage_session_pool50_ids`, `stage_session_final_ids`
- dump-only maps in `write_one_replay_dump` and `dump_turn_ic_novel` (line 6884) are `BTreeMap` / `BTreeSet`. `HashMap::new` there would advance the thread-local seed between queries.

`benchmarks/score_board.mjs` reads those fields after `rank()` returns. LoCoMo dialogues 6–10 are not loaded unless `FREEZE=1`.

What each line is:

- gold in corpus: at least one gold id was in the indexed sessions. Turn gold for LME is `has_answer` from the oracle, joined after ranking.
- gold in pool@50: at least one gold id is in that top-50 list. This is before submodular selection. It is not the harness window.
- gold at pre-Kemeny@5: at least one gold id is in the first 5 of the already-selected window (length about `top_k`, after submodular selection, ReplySource, boundary, given-new, opener, hearsay, and `rsb_seat_behind`). Not `rank_pre`.
- gold at post-Kemeny@5: first 5 after `rank1_gatekeeper_keep_rsb_head`. This list equals the returned list on every scored question (`post_matches_returned` 100%).
- gold at final @1: position 0 of that same list.
- Sessions have no Kemeny pass. Session pre-Kemeny@5 and post-Kemeny@5 are both `stage_session_final_ids`.

`order` tie-break is higher score, then newer session timestamp, then lower index.

### Board before = after

Instrumented binary sha256 `3c85ba00233d6871abd779478e1685ac22fbb5495fdb4507684ab4339e96b875`. Compared with `rankings-seed1.jsonl`: 1854 lists (500 LME + LoCoMo tune + 355 BEAM), 0 diffs. Rates unchanged from Phase 0. LoCoMo test dialogues were not loaded.

### Funnel, any-hit, tune split for LoCoMo

| slice | n | corpus | pool@50 | pre-Kemeny@5 | post-Kemeny@5 | final@1 |
|---|---:|---:|---:|---:|---:|---:|
| LME session | 470 | 470 | 470 | 463 | 463 | 428 |
| LME turn | 470 | 470 | 462 | 386 | 396 | 300 |
| LoCoMo tune session | 997 | 997 | 997 | 937 | 937 | 776 |
| LoCoMo tune turn | 997 | 997 | 941 | 705 | 754 | 510 |
| tune Multi-Hop turn | 142 | 142 | 123 | 76 | 77 | 46 |
| tune Open-Domain turn | 44 | 44 | 31 | 15 | 17 | 10 |
| tune Temporal turn | 156 | 156 | 150 | 127 | 132 | 108 |
| tune Single-Hop turn | 418 | 418 | 405 | 306 | 328 | 222 |
| tune Adversarial turn | 237 | 237 | 232 | 181 | 200 | 124 |
| BEAM session | 355 | 355 | 343 | 247 | 247 | 128 |
| BEAM turn | 355 | 355 | 324 | 215 | 222 | 148 |

One LoCoMo tune question has a gold turn id that is not indexed (`corpus_all` 996/997). Every other slice has `corpus_all == n`. No gold id that reached final@1 was outside pool@50.

Multi-Hop tune: 19/142 gold turns are absent from pool@50. 123 are in the pool and 77 are in post-Kemeny@5. The larger bucket is found-but-ranked-below-5, not never-found. Open-Domain tune: 13/44 absent from pool@50, 31 in the pool, 17 in post-Kemeny@5. Both buckets are large. Session gold for both categories is entirely inside pool@50.

Kemeny never changes the session funnel. On turns it raises the @5 hit (LME 386 to 396, LoCoMo tune 705 to 754, BEAM 215 to 222).

### One question

LME `e47becba`, "What degree did I graduate with?". Gold session `answer_280352e9` is in the session pool (36 sessions scored), in pre-Kemeny@5, in post-Kemeny@5, and at final@1. Gold turns (2) are in the turn pool of 50, in both @5 windows, and at final@1. Returned turns start `answer_280352e9#4`, `#5`, `#3`, `#2`, `d94b721b#8`. Same as both Phase 0 seeds.

## R3 — drop intent cue lists

Commit `6f10db31254665932129b0c62fa9e4a231c4e6d8`. Symbols in `crates/vestige-core/src/retrieval.rs`:

- deleted `PREF_QUERY_CUES`, `ASSISTANT_ATTRIBUTION_CUES`, `COLLECTIVE_CHAT_CUES`, `INFERENCE_CUES`, `STATE_REVISION_CUES`
- deleted `TIMELINE_CUES`, whose only caller was the preference suppressor. `is_event_sequence` still carries those phrases
- `classify` (line 3879): `preference`, `assistant_attribution`, and `inference` are false. `user_attribution` is the closed-class pronoun list only. `plan_query` can still set `preference` from a constraint term the haystack attested
- `rank_one` (line 4420): removed `persona_infer_on` and `episodic_user_quote` and both 1.38/1.40 branches. Question-echo demotion is no longer skipped for those queries
- `rank_standing_rules` (line 4294): dropped the episodic-quote early return
- supersession `has_rev_cue` (line 3449) is `is_personal_constraint` only
- tests renamed to match: `assistant_and_preference_phrasing_does_not_set_intent`, `beam_timeline_routes_to_user_and_collective_phrasing_is_not_assistant`, `preference_phrasing_does_not_promote_the_prefer_turn`

`cargo test -p vestige-core --lib retrieval`: 88 passed.

Binary sha256 `0612286d6bcadcde3b0b271652f0605668d1b23797465116f77125488c3fc066`.

### Board

| slice | before | after | delta |
|---|---|---|---|
| LME session Any@1/5/10 | 428/463/467 | 426/464/466 | -2 / +1 / -1 |
| LME session All@5 | 420 | 421 | +1 |
| LME turn Any@1/5/10 | 300/396/417 | 269/396/419 | -31 / 0 / +2 |
| LoCoMo tune Sess R@1/R@5 | 776/937 (77.83/93.98) | 785/940 (78.74/94.28) | +9 / +3 |
| LoCoMo tune Exact R@1/R@5/R@10 | 510/754/816 | 512/751/815 | +2 / -3 / -1 |
| LoCoMo tune Soft R@5 | 809 (81.14) | 812 (81.44) | +3 |
| BEAM session Any@1/5/10 | 128/247/276 | 127/239/272 | -1 / -8 / -4 |
| BEAM session All@5 | 143 | 136 | -7 |
| BEAM turn Any@1/5/10 | 148/222/254 | 141/212/244 | -7 / -10 / -10 |

LME single-session-assistant turn Any@1 went from 39/56 to 5/56. Session Any stayed 56/56/56. The gold session is still found; the assistant answer is no longer pulled to rank 1. LoCoMo tune Multi-Hop session R@1 72 to 75, turn Exact R@1 46 to 48, Exact R@5 77 to 76. Open-Domain session and exact rates did not move; Soft R@5 20 to 21.

This removal stays. Restoring the lists would put the benchmark phrasing back. A later structural pass has to earn the assistant-turn hits without those cues.

### Funnel delta (any-hit, after minus before)

| slice | corpus | pool@50 | pre-Kemeny@5 | post-Kemeny@5 | final@1 |
|---|---:|---:|---:|---:|---:|
| LME session | 0 | 0 | +1 | +1 | -2 |
| LME turn | 0 | 0 | -13 | 0 | -31 |
| LoCoMo tune session | 0 | 0 | +3 | +3 | +9 |
| LoCoMo tune turn | 0 | 0 | -3 | -3 | +2 |
| Multi-Hop turn | 0 | 0 | 0 | -1 | +2 |
| Open-Domain turn | 0 | 0 | 0 | 0 | 0 |
| BEAM session | 0 | 0 | -8 | -8 | -1 |
| BEAM turn | 0 | -5 | -6 | -10 | -7 |

Pool@50 did not move on LME or LoCoMo. The LME turn loss is inside the selected window: post-Kemeny@5 is unchanged at 396, final@1 drops 31.

### One question

LME `6ae235be` (single-session-assistant): "I remember you told me about the refining processes at CITGO's three refineries earlier. Can you remind me what kind of processes are used at the Lake Charles Refinery?"

Before: turn rank 1 is `answer_sharegpt_IUWQYGQ_0#1`, the assistant answer. After: turn rank 1 is `answer_sharegpt_IUWQYGQ_0#0`, the user question that asked for the refineries; the assistant answer sits at rank 2. Session rank 1 is `answer_sharegpt_IUWQYGQ_0` both times.

## R4 — delete the hypernym lattice

Commit `8bf10d2ad91aa8abd36cb0bcb7e6d53123c869f2`. Symbols in `crates/vestige-core/src/retrieval.rs`:

- deleted `HYPERNYM_LATTICE` (all 40 clusters, including VIDEO, IDENTITY, STORAGE, SECURITY, and the product words strata/precedent/salience)
- `hypernyms` (line 452) returns an empty map. `enable_hypernyms` stays on `Params` so older payloads still parse
- `common_speaker_tokens` no longer inserts lattice words
- `non_hub_domains` (line 3614) is an empty set. `domain_session_df` is gone
- test `sibling_words_do_not_bridge_through_a_domain_list` (line 7174): "sisters" does not match "siblings"

`cargo test -p vestige-core --lib retrieval`: 88 passed.

Binary sha256 `ca6b7edda71841e045fd712fa12b8cdf201c459fede8980afadb9a3d13ccbb50`.

### Board (before is R3)

| slice | before | after | delta |
|---|---|---|---|
| LME session Any@1/5/10 | 426/464/466 | 426/463/466 | 0 / -1 / 0 |
| LME session All@5 | 421 | 420 | -1 |
| LME turn Any@1/5/10 | 269/396/419 | 266/399/422 | -3 / +3 / +3 |
| LoCoMo tune Sess R@1/R@5 | 785/940 | 780/941 | -5 / +1 |
| LoCoMo tune Exact R@1/R@5/R@10 | 512/751/815 | 505/749/820 | -7 / -2 / +5 |
| LoCoMo tune Soft R@5 | 812 | 815 | +3 |
| BEAM session Any@1/5/10 | 127/239/272 | 124/241/274 | -3 / +2 / +2 |
| BEAM session All@5 | 136 | 135 | -1 |
| BEAM turn Any@1/5/10 | 141/212/244 | 138/210/244 | -3 / -2 / 0 |

Tune Multi-Hop session R@1 stayed 75/142. Turn Exact R@1 48 to 46, R@5 stayed 76, R@10 92 to 94. Open-Domain session R@1 20 to 17, R@5 30 to 29, Exact R@1 stayed 10/44, Exact R@5 17 to 16, R@10 20 to 17. The lattice was carrying some Open-Domain session hits. It stays deleted.

### Funnel delta (after minus R3)

| slice | pool@50 | pre-Kemeny@5 | post-Kemeny@5 | final@1 |
|---|---:|---:|---:|---:|
| LME session | 0 | -1 | -1 | 0 |
| LME turn | -3 | +3 | +3 | -3 |
| LoCoMo tune session | 0 | +1 | +1 | -5 |
| LoCoMo tune turn | -7 | +1 | -2 | -7 |
| Multi-Hop turn | -1 | -1 | 0 | -2 |
| Open-Domain turn | -4 | -1 | -1 | 0 |
| BEAM session | -3 | +2 | +2 | -3 |
| BEAM turn | +1 | 0 | -2 | -3 |

Corpus unchanged on every slice. Open-Domain's Exact R@1 loss is zero; four gold turns left pool@50.

### One question

LME `f8c5f88b`: "Where did I buy my new tennis racket from?"

Session rank 1 stays `answer_c3567066`. Turn rank 1 moves from `answer_c3567066#4` (the user turn that names the new racket and where it came from) to `answer_c3567066#6` (the user turn about buying tennis balls). The shared SPORTS domain no longer ties those turns together.

### One question

LME `e47becba` (`single-session-user`): "What degree did I graduate with?"

Gold session `answer_280352e9`. Seed 1 and seed 2 both return it at session rank 1, turn `answer_280352e9#4` at turn rank 1.

## R5 — delete preference and constraint marker lists

Commit `369f63cb8523179b1595a7b1f37f419c5511add2`. `crates/vestige-core/src/retrieval.rs`:

- deleted `PREF_MARKERS`, `HARD_CONSTRAINT_MARKERS`, `USER_SCHEMA_CUES`, `BLIP_STOPLIST`
- `user_preference_speech_act` (line 2543) is `hedged_like_is_preference || commitment_manner`. Those two helpers were not in the removal list and stay
- `build_index` (line 3051) hardwires `is_personal_constraint = false`. `is_user_schema` (line 3052) is field 0 and (`pref_marker` or `habitual_perfect`)
- BLIP caption tokens (line 2996) are indexed when `is_content`, with no caption stoplist
- `constraint_object_tokens` and the `plan_query` preference-from-constraint path stay. `is_structural_personal_turn` still has the `turn_pos <= 4` arm (R7)
- `has_rev_cue` still reads `is_personal_constraint`, which is now always false, so that supersession edge does not fire

`cargo test -p vestige-core --lib retrieval`: 88 passed. No test asserted the deleted lists.

Binary sha256 `cbd3aa11dcf2d71283844638840e30861fbb5cde101286cb9631d13e70f1c98e`.

Input check (`top_k=10`, gold = any-hit). LME turns returned length 10 / 10.02 / 11. Session gold in corpus 470, in pool@50 470, in the returned window 467, absent from the window 3. Turn gold in corpus 470, in pool@50 459, in the returned window 423, absent from the window 47. LoCoMo tune turns length 10. Turn gold in corpus 997 (all-of 996), in pool@50 934, in the returned window 824. BEAM turns length 1 / 10.14 / 11. Turn gold in corpus 355, in pool@50 320, in the returned window 246. `stage_turn_pre_kemeny_ids` is the selected window immediately before `rank1_gatekeeper_keep_rsb_head`, not a pre-rank list. `post_matches_returned` is 100% on every slice.

### Board (before is R4)

| slice | before | after | delta |
|---|---|---|---|
| LME session Any@1/5/10 | 426/463/466 | 426/464/467 | 0 / +1 / +1 |
| LME session All@5 | 420 | 422 | +2 |
| LME turn Any@1/5/10 | 266/399/422 | 266/398/423 | 0 / -1 / +1 |
| LoCoMo tune Sess R@1/R@5 | 780/941 | 781/941 | +1 / 0 |
| LoCoMo tune Exact R@1/R@5/R@10 | 505/749/820 | 504/750/824 | -1 / +1 / +4 |
| LoCoMo tune Soft R@5 | 815 | 816 | +1 |
| BEAM session Any@1/5/10 | 124/241/274 | 124/241/274 | 0 / 0 / 0 |
| BEAM session All@5 | 135 | 136 | +1 |
| BEAM turn Any@1/5/10 | 138/210/244 | 138/211/246 | 0 / +1 / +2 |

LME by type, only the rows that moved. Preference (n=30) session @5/@10 27/28 to 28/29, All@5 27 to 28, turns stayed 10/15/19. Multi-session All@5 97 to 98, turn @10 106 to 107. Temporal turn @5 107 to 106. User, knowledge-update, and assistant rows are unchanged (assistant turn @1 stays 5/56).

Tune Multi-Hop session R@1 75 to 74, R@5 127 to 126, Exact R@1 46 to 45, R@5 stayed 76, R@10 94 to 95, soft stayed 89. Open-Domain unchanged: session 17/29, Exact 10/16/17, soft 21. Single-Hop session 357/401 to 359/403, Exact 223/325/360 to 224/326/362, soft 356 to 357. Temporal session R@5 153 to 152, Exact R@1 105 to 104. Adversarial Exact R@10 210 to 211.

### Funnel delta (after minus R4)

| slice | pool@50 | pre-Kemeny@5 | post-Kemeny@5 | final@1 |
|---|---:|---:|---:|---:|
| LME session | 0 | +1 | +1 | 0 |
| LME turn | 0 | 0 | -1 | 0 |
| LoCoMo tune session | 0 | 0 | 0 | +1 |
| LoCoMo tune turn | 0 | -3 | +1 | -1 |
| Multi-Hop session | 0 | -1 | -1 | -1 |
| Multi-Hop turn | 0 | 0 | 0 | -1 |
| Open-Domain session | 0 | 0 | 0 | 0 |
| Open-Domain turn | 0 | 0 | 0 | 0 |
| BEAM session | 0 | 0 | 0 | 0 |
| BEAM turn | 0 | 0 | +1 | 0 |

Corpus unchanged. Pool@50 unchanged on every slice. The preference-marker removal did not drop gold out of the candidate pool.

### One question

LME `75832dbd` (`single-session-preference`): "Can you recommend some recent publications or conferences that I might find interesting?"

Gold session `answer_d87a6ef8` (the medical-image-analysis session, user turn 0 `has_answer`). Before, that session is outside the top 5 (`sharegpt_xni9yYO_6`, `sharegpt_E0YL5SX_157`, `ultrachat_382018`, `c7270a1b_1`, `ultrachat_165893`). After, it is session rank 5 and `ultrachat_382018` leaves the top 5. Turn rank 1 stays `sharegpt_xni9yYO_6#0`. Rank-1 did not move on any LME question (13 rank-1 flips total: 12 LoCoMo, 1 BEAM).

## R6 — NLTK stopwords, closed-class framing, closed-class causal cues

Commit `c14ca2a2221a043050dbdb31e104db6df14cd7f5`. `crates/vestige-core/src/retrieval.rs`:

- `STOPWORDS` (line 263) is the NLTK English list from `nltk_data` `stopwords/english` (198 entries). Apostrophe-split debris (`don`, `ll`, `re`, …) is not copied, because `raw_tokens` joins `n't` (`don't` → `dont`) and strips `'s/'ve/'re/'ll/'m/'d` to a stem already in the list. The joined forms (`dont`, `wont`, `isnt`, …) are that image. Template words (`conversation`, `mentioned`, `remind`, `recommend`, `suggest`, …) are gone
- `QUERY_FRAMING_WORDS` (line 281) keeps closed-class time words only: ago, across, day/days, earlier/earliest, later/latest, month/months, throughout, week/weeks, year/years
- `CAUSAL_WORDS` (line 397) keeps `because`, `since`, `so that`, `due to`, `in order`. `wanted`/`needed`/`decided`/`helped`/`inspired`/`reason`/`purpose`/`aim` are gone
- `word_boundary_cues_never_misfire_on_substrings` (line 7794) now locks that open-class `aim` does not set `has_causal` and whole-word `because` does
- `STARTER_STOP` is a different list and was not part of this removal

`cargo test -p vestige-core --lib retrieval`: 88 passed.

Binary sha256 `7ac999402c545704f32bf4059ab9a364ad30e599e789070462e83af03f17fa1e`.

Input check (`top_k=10`). LME turns returned 10 / 10.04 / 11. Session gold in corpus 470, pool@50 470, returned window 468, absent from the window 2. Turn gold in corpus 470, pool@50 459, returned window 423, absent 47. LoCoMo tune turns 6 / 10 / 10. Turn gold in corpus 997 (all-of 996), pool@50 938, returned window 829. BEAM turns 10 / 10.15 / 11. Turn gold in corpus 355, pool@50 313, returned window 239. `stage_turn_pre_kemeny_ids` is the selected window immediately before `rank1_gatekeeper_keep_rsb_head`. `post_matches_returned` is 100% on every slice.

### Board (before is R5)

| slice | before | after | delta |
|---|---|---|---|
| LME session Any@1/5/10 | 426/464/467 | 425/461/468 | -1 / -3 / +1 |
| LME session All@5 | 422 | 418 | -4 |
| LME turn Any@1/5/10 | 266/398/423 | 253/396/423 | -13 / -2 / 0 |
| LoCoMo tune Sess R@1/R@5 | 781/941 | 772/938 | -9 / -3 |
| LoCoMo tune Exact R@1/R@5/R@10 | 504/750/824 | 511/757/829 | +7 / +7 / +5 |
| LoCoMo tune Soft R@5 | 816 | 811 | -5 |
| BEAM session Any@1/5/10 | 124/241/274 | 128/230/262 | +4 / -11 / -12 |
| BEAM session All@5 | 136 | 133 | -3 |
| BEAM turn Any@1/5/10 | 138/211/246 | 136/207/239 | -2 / -4 / -7 |

LME turn @1 loss is inside the window: pool@50 stayed 459, pre-Kemeny@5 376 to 362. By type, temporal turn @1 84 to 78, multi-session turn @1 66 to 61, preference turn @1 10 to 8. Assistant turn @1 stays 5/56.

Tune Multi-Hop session 74/126 to 71/123, Exact 45/76/95 to 42/72/93, soft 89 to 85. Open-Domain session 17/29 to 14/30, Exact 10/16/17 to 8/17/21, soft 21 to 19. Single-Hop Exact 224/326/362 to 230/336/365. The removal stays. Recovery is not a word list.

### Funnel delta (after minus R5)

| slice | pool@50 | pre-Kemeny@5 | post-Kemeny@5 | final@1 |
|---|---:|---:|---:|---:|
| LME session | 0 | -3 | -3 | -1 |
| LME turn | 0 | -14 | -2 | -13 |
| LoCoMo tune session | 0 | -3 | -3 | -9 |
| LoCoMo tune turn | +4 | +16 | +7 | +7 |
| Multi-Hop turn | +1 | -5 | -4 | -3 |
| Open-Domain turn | +1 | +2 | +1 | -2 |
| BEAM session | +1 | -11 | -11 | +4 |
| BEAM turn | -7 | -7 | -4 | -2 |

### One question

LME `505af2f5` (`single-session-preference`): "I was thinking of trying a new coffee creamer recipe. Any recommendations?"

Turn rank 1 moves from `sharegpt_kjeGJvK_11#0` (an assistant list of coffee smoothie recipes) to `4f2d6be4_2#6` (a user turn: "I'm thinking of trying some new coffee flavors, do you have any recommendations…"). `trying` and `new` were query-framing tokens and are now content, so that user turn covers them. Gold session `answer_f3164f2c` stays session rank 3. Its gold turn is the user turn about almond-milk creamer.
