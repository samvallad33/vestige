# Per-question diff: both pool-A seats versus the Step 31+32+33+34 tip

Vectors are the `c2_bench.py` per-question rows.

- LME: `[S@1, S@5, S@10, T@1, T@5, T@10, TA@5]`
- LoCoMo: `[S@1, S@5, T@1, T@5, T@10]`
- BEAM: `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`

Losses on any coordinate, all three benches: **0**. The anaphoric tail versus the buried-peak commit alone is also 0 losses.

LME coordinate changes: 0. BEAM coordinate changes: 0. LME session-coordinate drops: 0.

## Gains

| Bench | Question | Seat | Tip | After |
| --- | --- | --- | --- | --- |
| LoCoMo | `9:137` | buried lexical peak | `[1, 1, 0, 0, 0]` | `[1, 1, 0, 1, 1]` |
| LoCoMo | `7:146` | anaphoric tail | `[0, 0, 0, 0, 1]` | `[0, 0, 0, 1, 1]` |
| LoCoMo | `9:113` | anaphoric tail | `[0, 1, 0, 0, 1]` | `[0, 1, 0, 1, 1]` |
| LoCoMo | `9:131` | anaphoric tail | `[0, 1, 0, 0, 1]` | `[0, 1, 0, 1, 1]` |
| LoCoMo | `9:146` | anaphoric tail | `[1, 1, 0, 0, 1]` | `[1, 1, 0, 1, 1]` |

Single-Hop Turn@5 686→691. Single-Hop Turn@10 737→738, from `9:137` only. Turn@1 and both session metrics stay put on every row.
