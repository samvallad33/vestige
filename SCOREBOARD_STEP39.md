# SCOREBOARD Step 39

Combined tip `f8570b9a` (Steps 31–38 + Pools C, B, A) versus Step 39 capture-recapture. Counts are from `retrieval_bench` with the replay dump off.

| Bench | Metric | Before | After | Delta |
| --- | --- | --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 | 0 |
| LME_S | Session All@5/All@10 | 424/447 | 424/447 | 0 |
| LME_S | Turn @1/@5/@10 | 355/429/438 | 355/429/438 | 0 |
| LME_S | Turn All@5 | 320 | 320 | 0 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 812/1134/1229 | 812/1136/1229 | @5 +2 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 | 0 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 162/240/269 | 162/240/269 | 0 |
| BEAM-100K | Turn All@5/All@10 | 122/142 | 122/142 | 0 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 | 0 |

LoCoMo Turn@5 by category: Multi-Hop 158, Temporal 250→251, Open-Domain 35, Single-Hop 691→692.

Per-question gains: LoCoMo `6:15` Turn@5, LoCoMo `8:109` Turn@5. Losses: none.

90% need, not met by this step: LME Turn@1 423, LME Turn All@5 423, LoCoMo Turn@1 1383, BEAM Turn@1 320, BEAM All@5 320, BEAM Session@1 320.
