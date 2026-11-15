# Per-question diff vs bd2693e2 baseline

Compared `baseline-bd2693e2.json` to the Step 31 run (`step31.json`) on every stored bit.

- LME_S (470): S@1 S@5 S@10 T@1 T@5 T@10 TA@5. One gain, zero losses.
- LoCoMo (1536): S@1 S@5 T@1 T@5 T@10. Zero gains, zero losses.
- BEAM-100K (355): T@1 T@5 T@10 All@5 S@1 S@5 SAll@5. Zero gains, zero losses.

## Gain

| bench | question | type | bit | before | after |
| --- | --- | --- | --- | ---: | ---: |
| LME_S | `e8a79c70` | single-session-assistant | Turn @1 | 0 | 1 |

Question: "I was going through our previous conversation about making a classic French omelette, and I wanted to confirm - how many eggs did you say we need for the recipe?"

Gold turn `answer_ultrachat_13075#1` (assistant). It was inside the top 5 (Turn @5 already 1) and is now slot 1. Turn @5, Turn @10, Turn All@5, and every session bit on this question stayed 1.

## Losses

None.
