# SCOREBOARD W2-001

WIN. Combined Step 31–36 tip versus `apply_c2_w2001_session_head`.

Baseline tip is the replayed stack ending at Step 36 (`1a67f43f`), measured before this change. LME Turn@1 on that combined tip is 354. The uncombined floor in the brief was 353; every listed historical Turn@1 flip still hits (`e8a79c70`, `c4a1ceb8`, `ccb36322`, `gpt4_2ba83207`, the LoCoMo ids, and the BEAM ids).

| Bench | Metric | Baseline | W2-001 | Delta |
| --- | --- | --- | --- | --- |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 432/463/468 | +2/0/0 |
| LME_S | Session All@5/All@10 | 424/447 | 424/447 | 0/0 |
| LME_S | Turn @1/@5/@10 | 354/428/437 | 354/428/437 | 0/0/0 |
| LME_S | Turn All@5 | 314 | 314 | 0 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 811/1127/1226 | 811/1127/1226 | 0/0/0 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1138/1416 | +5/0 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 161/238/267 | 161/238/267 | 0/0/0 |
| BEAM-100K | Turn All@5/All@10 | 118/138 | 118/138 | 0/0 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 140/249/275 | +7/0/0 |
| BEAM-100K | Session All@5/All@10 | 137/178 | 137/178 | 0/0 |

Per-question losses versus the combined tip: 0 on every stored cell (LME 7 cells, LoCoMo 5, BEAM 7). Session @5/@10 sets are unchanged because a promotion only rotates a session that is already inside the top 5. Turn lists are not written.

Category Session@1 (only cells that moved):

- LME single-session-user 62→63, single-session-preference 17→18. knowledge-update stayed 71.
- LoCoMo Single-Hop 699→703, Temporal 242→243. Multi-Hop 156 and Open-Domain 36 stayed.
- BEAM information_extraction 15→18, knowledge_update 24→27, summarization 6→7. No BEAM category dropped.

Questions that flipped Session@1 from miss to hit:

- LME: `5d3d2817`, `d6233ab6`
- LoCoMo: `2:54`, `3:88`, `4:110`, `4:177`, `8:94`
- BEAM: `2:information_extraction:1`, `7:information_extraction:0`, `7:knowledge_update:1`, `7:summarization:1`, `8:knowledge_update:1`, `10:knowledge_update:0`, `12:information_extraction:1`
