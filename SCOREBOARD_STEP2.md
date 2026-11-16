# SCOREBOARD Step 2

Binary: `target/release/retrieval_bench`. Harness: `tools/c2_bench.py`. Gate questions only. Step 1 is `e81a99b2` plus the measurement dump and no ranker change. Step 2 is that tree plus the nominal terminal seat.

Session pre_rank@5 = session post_kemeny@5 = final session @5, because `final_session_indices` is immutable.

## LME_S n=470

| | Step 1 | Step 2 |
|---|---:|---:|
| Turn any@1 | 360 | 360 |
| Turn any@5 | 428 | 428 |
| Turn any@10 | 441 | 441 |
| Turn all@5 | 323 | 323 |
| Session any@1 | 430 | 430 |
| Session any@5 | 463 | 463 |
| Session any@10 | 469 | 469 |
| Session all@5 | 422 | 422 |
| Session all@10 | 446 | 446 |
| Turn corpus / pool@50 / pre_rank@5 / post_kemeny@5 / @1 | 470 / 464 / 406 / 415 / 360 | same |
| Session corpus / pool@50 / pre_rank@5 / post_kemeny@5 / @1 | 470 / 470 / 463 / 463 / 430 | same |

`c2_bench.py --diff`: losses=0 gains=0. Session drops: 0.

## LoCoMo n=1536

| | Step 1 | Step 2 |
|---|---:|---:|
| Turn any@1 | 812 | 816 |
| Turn any@5 | 1130 | 1130 |
| Turn any@10 | 1236 | 1236 |
| Session any@1 | 1137 | 1137 |
| Session any@5 | 1422 | 1422 |
| Turn corpus / pool@50 / pre_rank@5 / post_kemeny@5 / @1 | 1535 / 1424 / 1039 / 1100 / 812 | 1535 / 1424 / 1039 / 1100 / 816 |
| Session corpus / pool@50 / pre_rank@5 / post_kemeny@5 / @1 | 1536 / 1536 / 1422 / 1422 / 1137 | same |

`c2_bench.py --diff`: losses=0 gains=4, all turn @1.

| question | cat | Step 1 head | Step 2 head | gold |
|---|---|---|---|---|
| conv-26:103 | 4 | D6:9 | D6:10 | D6:10 |
| conv-30:57 | 4 | D7:4 | D7:5 | D7:5 |
| conv-48:41 | 1 | D11:6 | D28:18 | D11:4, D11:7, D28:16, D28:18 |
| conv-49:131 | 4 | D18:7 | D18:8 | D18:8 |

Top-5 membership is unchanged on each of the four. `conv-50:69` cites evidence `D30:05`; the dialogue id is `D30:5`. Gold-in-corpus stays 1535/1536 on Step 0, Step 1, and Step 2.

## BEAM-100K n=355

| | Step 1 | Step 2 |
|---|---:|---:|
| Turn any@1 | 164 | 164 |
| Turn any@5 | 241 | 241 |
| Turn any@10 | 272 | 272 |
| Turn all@5 | 125 | 125 |
| Turn all@10 | 146 | 146 |
| Session any@1 | 133 | 133 |
| Session any@5 | 249 | 249 |
| Session any@10 | 275 | 275 |
| Turn corpus / pool@50 / pre_rank@5 / post_kemeny@5 / @1 | 355 / 327 / 229 / 234 / 164 | same |
| Session corpus / pool@50 / pre_rank@5 / post_kemeny@5 / @1 | 355 / 341 / 249 / 249 / 133 | same |

`c2_bench.py --diff`: losses=0 gains=0.

## Pre-flight and Step 0, same harness

Pre-flight `894c7d44` and Step 0 `45fa0769`, measurement dump only: 2361 gate questions, losses=0 gains=0. LME and BEAM match the table's Step 1 column, funnel included. LoCoMo turn any@1/@5/@10 = 809/1128/1234, session any@1/@5 = 1133/1420. Turn funnel 1535/1422/1037/1098/809. Session funnel 1536/1536/1420/1420/1133. Step 1's LoCoMo gains against that floor are `conv-47:29`, `conv-47:31`, `conv-48:19`, `conv-48:68`, losses=0.
