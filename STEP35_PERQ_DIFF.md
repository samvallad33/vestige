# Per-question diff, Step 35 vs Step 31+32+33 tip

Vectors are the `c2_bench.py` per-question bits.

- LME: `[s1, s5, s10, t1, t5, t10, ta5]`
- LoCoMo: `[s1, s5, t1, t5, t10]`
- BEAM: `[t1, t5, t10, all5, s1, s5, sall5]`

Compared on all 470 LME + 1536 LoCoMo + 355 BEAM questions. One vector changed. Zero bits went from 1 to 0.

| bench | key | tip | step 35 | change |
|---|---|---|---|---|
| LME | `gpt4_2ba83207` | `[1, 1, 1, 0, 1, 1, 0]` | `[1, 1, 1, 1, 1, 1, 0]` | Turn@1 0→1 |

Question: "Which grocery store did I spend the most money at in the past month?" Gold turns include `answer_6a3b5c13_4#6`, which Step 35 seats at slot 1. Session@1 was already 1.
