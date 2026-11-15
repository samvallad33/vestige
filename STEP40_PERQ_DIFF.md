# Step 40 per-question diff vs the combined 31–38 + Pools C/B/A tip

Baseline replay tip `9d95cc42a6778cc5e044c6a244c2fe12c714c915`, binary sha256 `dd449987923b653de2423edb495553da47d9ee005a5a5d2ea6b875be4ff1afbf`.
After binary sha256 `59459169f862745acb45b88459c604fb41feba8b9ed5fcbd772ffcf6ba38c350`.

Vectors are the `c2_bench.py` bits.

- LME: `[S@1, S@5, S@10, T@1, T@5, T@10, TAll@5]`
- LoCoMo: `[S@1, S@5, T@1, T@5, T@10]`
- BEAM: `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`

LME: 0 questions differ (470 compared).
BEAM: 0 questions differ (355 compared).
LoCoMo: 1 question differs, 0 losses (1536 compared).

| question | before | after |
|---|---|---|
| LoCoMo `4:74` | `[1,1,0,1,1]` | `[1,1,1,1,1]` |

Slot 1 before is `D1:3`. Slot 1 after is `D1:16` (gold). Both turns have scene score `297.7453308105469` (identical bits). `D1:16` was already inside the top 5, so Turn@5 and Turn@10 stay 1. Session bits stay `[1,1]`. Top-5 after the swap: `D1:16`, `D1:3`, `D1:2`, `D1:13`, `D22:2`.
