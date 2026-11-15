# Scoreboard Step 35

Tip is Step 31 + Step 32 + Step 33 (`5ffa0fdd`). Step 35 is Mill's method of agreement. One per-question change on the three benches: LME `gpt4_2ba83207` Turn@1 0→1. No losses.

## LME_S (n=470, Turn n=470)

| metric | tip | step 35 |
|---|---:|---:|
| Session any@1/5/10 | 430/463/468 | 430/463/468 |
| Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| Turn any@1/5/10 | 352/428/437 | 353/428/437 |
| Turn All@5 | 314 | 314 |
| multi-session Turn@1 | 89/121 | 90/121 |
| single-session-user Turn@1 | 54/64 | 54/64 |
| single-session-assistant Turn@1 | 42/56 | 42/56 |
| single-session-preference Turn@1 | 12/30 | 12/30 |
| temporal-reasoning Turn@1 | 93/127 | 93/127 |
| knowledge-update Turn@1 | 62/72 | 62/72 |

## LoCoMo (n=1536)

| metric | tip | step 35 |
|---|---:|---:|
| Turn any@1/5/10 | 804/1127/1226 | 804/1127/1226 |
| Session any@1/5 | 1133/1416 | 1133/1416 |
| Multi-Hop Turn@1 | 97/282 | 97/282 |
| Open-Domain Turn@1 | 20/92 | 20/92 |
| Temporal Turn@1 | 189/321 | 189/321 |
| Single-Hop Turn@1 | 498/841 | 498/841 |

## BEAM-100K (n=355)

| metric | tip | step 35 |
|---|---:|---:|
| Turn any@1/5/10 | 159/238/267 | 159/238/267 |
| Turn All@5 / All@10 | 118 / 138 | 118 / 138 |
| Session any@1/5/10 | 133/249/275 | 133/249/275 |
| multi_session_reasoning Turn@1 | 21/40 | 21/40 |
| temporal_reasoning Turn@1 | 29/40 | 29/40 |
| knowledge_update Turn@1 | 31/40 | 31/40 |

Per-question vectors: `STEP35_PERQ_DIFF.md`.
