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

Not measured on this binary. `rank_pre` is not pre-rank. Stage ids are added in the next commit, which must not move this board.

### One question

LME `e47becba` (`single-session-user`): "What degree did I graduate with?"

Gold session `answer_280352e9`. Seed 1 and seed 2 both return it at session rank 1, turn `answer_280352e9#4` at turn rank 1.
