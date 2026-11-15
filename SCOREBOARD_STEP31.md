# SCOREBOARD Step 31 Exchequer

Baseline is the am tip of Computer 2's 44 (`e495b881`, tree `4612c06f`), scores identical to the published bd2693e2 board. After is `retrieval_bench` sha256 `d01b44eb1611b3bb887d3e0f27faaf831ab95c5c8f36f3e9e65a7602ddb64373`.

## LME_S n=470

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| Session @1 | 430 | 430 | 0 |
| Session @5 | 463 | 463 | 0 |
| Session @10 | 468 | 468 | 0 |
| Session All@5 | 424 | 424 | 0 |
| Session All@10 | 447 | 447 | 0 |
| Turn @1 | 350 | 351 | +1 |
| Turn @5 | 428 | 428 | 0 |
| Turn @10 | 437 | 437 | 0 |
| Turn All@5 | 314 | 314 | 0 |
| single-session-assistant Turn @1 | 41 | 42 | +1 |

## LoCoMo n=1536

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| Turn @1 | 796 | 796 | 0 |
| Turn @5 | 1127 | 1127 | 0 |
| Turn @10 | 1226 | 1226 | 0 |
| Session @1 | 1133 | 1133 | 0 |
| Session @5 | 1416 | 1416 | 0 |
| Multi-Hop Turn @1 | 97 | 97 | 0 |
| Temporal Turn @1 | 187 | 187 | 0 |
| Open-Domain Turn @1 | 18 | 18 | 0 |
| Single-Hop Turn @1 | 494 | 494 | 0 |

## BEAM-100K n=355

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| Turn @1 | 156 | 156 | 0 |
| Turn @5 | 238 | 238 | 0 |
| Turn @10 | 267 | 267 | 0 |
| All@5 | 118 | 118 | 0 |
| All@10 | 138 | 138 | 0 |
| Session @1 | 133 | 133 | 0 |
| Session @5 | 249 | 249 | 0 |
| Session @10 | 275 | 275 | 0 |
| instruction_following Turn @1 | 6 | 6 | 0 |
| instruction_following Turn @5 | 18 | 18 | 0 |
| instruction_following All@5 | 13 | 13 | 0 |

Per-question losses: 0 on all three benchmarks. The only bit that changed is LME `e8a79c70` Turn @1, 0 → 1.
