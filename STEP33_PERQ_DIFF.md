# Step 33 per-question diff vs Step 31 tip

Compared to the Step 31 per-question json (am tip `99e9d44738d440aabab5d1dc6198f523a2f277a5`).

LME_S, 470 questions, bits S@1 S@5 S@10 T@1 T@5 T@10 TurnAll@5: +0 −0 on every bit.

BEAM-100K, 355 questions, bits T@1 T@5 T@10 All@5 S@1 S@5 SessionAll@5: +0 −0 on every bit.

LoCoMo, 1536 questions: S@1 +0 −0, S@5 +0 −0, T@5 +0 −0, T@10 +0 −0. T@1 +3 −0.

| key | category | question | gold | baseline slot 1 | after slot 1 | T@1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4:28 | Open-Domain | Which popular music composer's tunes does Tim enjoy playing on the piano? | D8:14, D8:16 | D21:13 | D8:14 | 0 → 1 |
| 4:44 | Temporal | When did John and his wife go on a European vacation? | D16:14 | D17:3 | D16:14 | 0 → 1 |
| 8:124 | Single-Hop | What does Evan mention about his progress at the gym to Sam? | D14:8 | D4:16 | D14:8 | 0 → 1 |

Each gold turn was already inside the top 5. It was the strict unique maximum of `t_own` and of `s_score + bridge_boost`, and its query-term set strictly covered the baseline head. Top-5 membership did not change.
