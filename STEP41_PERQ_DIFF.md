# Step 41 per-question diff vs the combined pack-43 tip

Baseline replay tip `1a0794b49e67e020f4540d7b727ce11e141d6097`, binary sha256 `911ffb8951c0db69903dc060c340f8e1702dafc7133aafce59537d1464beaede`.
After binary sha256 `8fea3e4702c388c5bedaadba1b9a1b8e2c5e86d4fcbb8b34bdca8892da6ff28d`.

Vectors are the `c2_bench.py` bits.

- LME: `[S@1, S@5, S@10, T@1, T@5, T@10, TAll@5]`
- LoCoMo: `[S@1, S@5, T@1, T@5, T@10]`
- BEAM: `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`

LME: 0 questions differ (470 compared).
BEAM: 0 questions differ (355 compared).
LoCoMo: 4 questions differ, 0 losses (1536 compared).

| question | before | after |
|---|---|---|
| LoCoMo `0:68` | `[1,1,0,1,1]` | `[1,1,1,1,1]` |
| LoCoMo `6:23` | `[1,1,0,1,1]` | `[1,1,1,1,1]` |
| LoCoMo `7:3` | `[1,1,0,1,1]` | `[1,1,1,1,1]` |
| LoCoMo `7:49` | `[1,1,0,1,1]` | `[1,1,1,1,1]` |

Each gold turn was already inside the top 5. The swap exchanges slot 1 with that turn. Session bits stay `[1,1]`.

- `0:68` Temporal, "How long has Melanie been practicing art?" Gold `D16:8`. Slot 1 before: `D16:7` Caroline, "How long have you been into art?" Slot 1 after: `D16:8` Melanie, "Seven years now, and I've finally found my real muses: painting and pottery." Top 5 after: `D16:8`, `D16:7`, `D16:5`, `D16:6`, `D8:6`. Sessions: `D16`, `D8`, `D17`.
- `6:23` Temporal, "When will John start his new job?" Gold `D13:5`. Slot 1 before: `D13:4` James, "When do you start?" Slot 1 after: `D13:5` John, "I'm starting next month." Top 5 after: `D13:5`, `D13:4`, `D13:3`, `D26:1`, `D13:6`. Sessions: `D13`, `D6`, `D18`.
- `7:3` Temporal, "When did Jolene's mother pass away?" Gold `D1:6`. Slot 1 before: `D1:5` Deborah, "she passed away a few years ago." Slot 1 after: `D1:6` Jolene, "My mother also passed away last year." Top 5 after: `D1:6`, `D1:5`, `D22:23`, `D1:7`, `D2:1`. Sessions: `D1`, `D22`, `D2`.
- `7:49` Single-Hop, "How long has Jolene been doing yoga and meditation?" Gold `D13:17`. Slot 1 before: `D13:16` Deborah, "How long have you been doing yoga and meditation?" Slot 1 after: `D13:17` Jolene, "about 3 years now." Top 5 after: `D13:17`, `D13:16`, `D20:21`, `D17:2`, `D13:15`. Sessions: `D13`, `D20`, `D7`.
