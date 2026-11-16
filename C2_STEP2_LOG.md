# C2 Step 2 attempt log

Floor: Step 1 tip `e81a99b2`, measured by `tools/c2_bench.py` before any ranker edit. LME_S n=470, LoCoMo n=1536, BEAM-100K n=355. Full counts are in `EXPERIMENT_LOG.md` and `SCOREBOARD_STEP2.md`.

A loss is any TARGET hit-flag that goes from 1 to 0 on a gate question. A top-5 reorder cannot change turn any@5, any@10, all@5, all@10, or any session metric.

## Attempt 0 — attached WIP, not applied

`step2-wip.patch` plus `multihop_overwrite.rs` from the pack. Rejected before a bench run. The module uses `std::collections::HashMap`, `classify_attr_pred` / action / Allen phrase tables (domain lists), and floating-point score scalers (1.25, 0.25, 1.15, +3, +6). House rules forbid all three. Not committed.

## Attempt 1 — numeric carrier overwrite, simulation on the Step 1 dump

Later of two numeric-carrier turns supersedes the earlier when the query names the carrier token, a wh-word is present, and interval cues are absent. Successor must already sit in the top 5.

BEAM: fires=9, gains=0, losses=5. Losses: `2:information_extraction:0`, `6:knowledge_update:1`, `11:information_extraction:0`, `11:multi_session_reasoning:0`, `15:multi_session_reasoning:1`. Rejected. Not implemented.

## Attempt 2 — broad `is_question` named-declarative seat, LoCoMo simulation

Head uses the engine `is_question` flag (a `?` anywhere the detector accepts, including a short prose with a trailing marker). Unique other-speaker nominal in the top 5 moves to slot 0.

LoCoMo gate: fire=20, gain=5, loss=1. Loss: `conv-49:141` cat 4, which was already turn @1. Cat 5 also loses `conv-48:227`. Rejected.

## Attempt 3 — BEAM top-5 stumps, simulation on the Step 1 dump

Unique digit-bearing turn that shares a content token with the query: fire=22, gain=2, loss=6. Later digit with two shared tokens: fire=9, gain=0, loss=3. Assistant-question to unique user declarative: fire=1, gain=0, loss=0 (no uplift). Strict unique query-overlap: fire=151, gain=7, loss=55. Rejected. None of these is a 0-loss BEAM Turn@1 rule.

## Attempt 4 — terminal-`?` nominal seat, landed

Head prose ends on `?` after the code fence is cut and trailing whitespace is stripped. This is stricter than `is_question`. Exactly one later top-5 turn is not a terminal question, its speaker is not `user` or `assistant`, its speaker differs from the head, and that speaker is a query token after a trailing `'s` / `'` is stripped. Move that turn to slot 0.

Simulation on the Step 1 LoCoMo dump: fire=10, gain=4, loss=0, including category 5. BEAM and LME roles are `user` / `assistant`, so the seat does not fire there.

Real `retrieval_bench` (release, funnel dump on), `tools/c2_bench.py --diff` against the Step 1 jsonl:

- LME_S 470: losses=0 gains=0
- LoCoMo 1536: losses=0 gains=4 (`conv-26:103` cat 4, `conv-30:57` cat 4, `conv-48:41` cat 1, `conv-49:131` cat 4), each turn @1 only
- BEAM 355: losses=0 gains=0

LoCoMo Turn@1 812/1536 → 816/1536. Every other TARGET metric is unchanged. This is the shipped rule.
