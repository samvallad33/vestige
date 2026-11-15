# Per-question diff: Step 34 versus the Step 31+32 tip

Baseline file: combined tip `3345f79f` (`/home/ubuntu/c2-runs/scores/tip31_32.json` on the measuring machine).
Candidate: Step 34 Bertillonage.

Vectors are the `c2_bench.py` per-question rows.

- LME: `[S@1, S@5, S@10, T@1, T@5, T@10, TA@5]`
- LoCoMo: `[S@1, S@5, T@1, T@5, T@10]`
- BEAM: `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`

Losses on any coordinate, all three benches: **0**.

## Gains

| Bench | Question | Tip | Step 34 |
| --- | --- | --- | --- |
| LME | `ccb36322` | `[0, 1, 1, 0, 1, 1, 1]` | `[0, 1, 1, 1, 1, 1, 1]` |
| LoCoMo | `1:42` | `[1, 1, 0, 1, 1]` | `[1, 1, 1, 1, 1]` |
| LoCoMo | `5:105` | `[1, 1, 0, 1, 1]` | `[1, 1, 1, 1, 1]` |
| LoCoMo | `9:9` | `[1, 1, 0, 1, 1]` | `[1, 1, 1, 1, 1]` |
| LoCoMo | `9:36` | `[1, 1, 0, 1, 1]` | `[1, 1, 1, 1, 1]` |
| BEAM | `19:multi_session_reasoning:0` | `[0, 1, 1, 0, 0, 1, 1]` | `[1, 1, 1, 0, 0, 1, 1]` |

On each row the only coordinate that changes is Turn @1. LME `ccb36322` stays Session @1 = 0; the seat does not rewrite the session list.

## Prior wins still hitting

| Question | Tip T@1 | Step 34 T@1 |
| --- | --- | --- |
| LME `e8a79c70` (Step 31) | 1 | 1 |
| LME `c4a1ceb8` (Step 32) | 1 | 1 |
| LoCoMo `2:106`, `3:143`, `5:13`, `6:25`, `9:147` | 1 | 1 |
| BEAM `9:temporal_reasoning:1`, `11:knowledge_update:0`, `20:temporal_reasoning:1` | 1 | 1 |
