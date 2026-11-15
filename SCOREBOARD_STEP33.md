# SCOREBOARD Step 33 Triangulation

Baseline is the Step 31 tip `99e9d44738d440aabab5d1dc6198f523a2f277a5` (tree `ed6d9d0b8cd9b606f6bbe8f5b7cb65d97c95e0a3`), scores identical to the published Step 31 board. After is `retrieval_bench` sha256 `9cc1a7ac610adc7f5e0ad7996d74ebd659512df3f5703326bdcb262a6c0a8b0b`.

## LME_S n=470

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| Session @1 | 430 | 430 | 0 |
| Session @5 | 463 | 463 | 0 |
| Session @10 | 468 | 468 | 0 |
| Session All@5 | 424 | 424 | 0 |
| Session All@10 | 447 | 447 | 0 |
| Turn @1 | 351 | 351 | 0 |
| Turn @5 | 428 | 428 | 0 |
| Turn @10 | 437 | 437 | 0 |
| Turn All@5 | 314 | 314 | 0 |

## LoCoMo n=1536

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| Turn @1 | 796 | 799 | +3 |
| Turn @5 | 1127 | 1127 | 0 |
| Turn @10 | 1226 | 1226 | 0 |
| Session @1 | 1133 | 1133 | 0 |
| Session @5 | 1416 | 1416 | 0 |
| Multi-Hop Turn @1 | 97 | 97 | 0 |
| Temporal Turn @1 | 187 | 188 | +1 |
| Open-Domain Turn @1 | 18 | 19 | +1 |
| Single-Hop Turn @1 | 494 | 495 | +1 |

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

Per-question losses: 0 on all three benchmarks. The only bits that changed are LoCoMo `4:28`, `4:44`, and `8:124` Turn @1, 0 → 1.
