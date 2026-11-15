# Scoreboard — pool A, buried lexical peak

Baseline is the combined Step 31 + 32 + 33 + 34 tip (this branch before the seat). That tip was measured once, together, before the edit. Its tree is the algo-44 tree `4612c06f` (the published bd2693e2 board) plus Steps 31, 32, 33 and 34. Algo-44 versus the tip: 0 per-question coordinate losses on LME, LoCoMo and BEAM (3 / 12 / 4 coordinate gains). Every listed Step 31–34 flip is still a Turn@1 hit on the tip.

After is one `retrieval_bench` release run of `apply_c2_buried_lexical_peak`.

## Headline

| Bench | Metric | Tip 31+32+33+34 | After |
| --- | --- | ---: | ---: |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 353/428/437 | 353/428/437 |
| LME_S | Turn All@5 | 314 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 808/1127/1226 | 808/1128/1227 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 160/238/267 | 160/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 | 118 / 138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 |

LoCoMo Turn@5 1127→1128 and Turn@10 1226→1227. Every other headline cell is unchanged. Per-question losses versus the tip: 0 on every metric of all three benches. LME Session drops: 0.

## LoCoMo by category

| Category | n | Tip T@1/@5/@10 | After T@1/@5/@10 | Tip S@1/@5 | After S@1/@5 |
| --- | ---: | --- | --- | --- | --- |
| Multi-Hop | 282 | 97/158/186 | 97/158/186 | 156/246 | 156/246 |
| Temporal | 321 | 191/248/260 | 191/248/260 | 242/299 | 242/299 |
| Open-Domain | 92 | 20/35/43 | 20/35/43 | 36/65 | 36/65 |
| Single-Hop | 841 | 500/686/737 | 500/687/738 | 699/806 | 699/806 |

## Question that moved

LoCoMo `9:137` (Single-Hop), "When did Calvin first get interested in cars?", evidence `D26:6`.

| | S@1 | S@5 | T@1 | T@5 | T@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tip | 1 | 1 | 0 | 0 | 0 |
| After | 1 | 1 | 0 | 1 | 1 |

`D26:6` is the unique named-speaker declarative whose `t_own` beats every turn in the scene pool of 50. It takes the lowest-own question seat in slots 1–4. Slot 0 stays `D2:4`, so Turn@1 does not move. The session list is untouched, so Session@1 stays 1 (`D26`).

## Gates

`cargo clippy -p vestige-core --all-targets -- -D warnings` clean. `cargo check -p vestige-mcp` clean. `cargo test -p vestige-core` 564 passed. Hash-check 0/500 on ranked ids and 0/500 on full responses.

## Stage funnel after the seat

Measured with `tools/c2_funnel.py`. Columns are gold in corpus / pool@50 / scene-rank 0 / final rank 0 / final top 5.

| Category | Tip | After |
| --- | --- | --- |
| Multi-Hop | 282 / 243 / 72 / 97 / 158 | 282 / 243 / 72 / 97 / 158 |
| Temporal | 320 / 299 / 182 / 191 / 248 | 320 / 299 / 182 / 191 / 248 |
| Open-Domain | 92 / 63 / 14 / 20 / 35 | 92 / 63 / 14 / 20 / 35 |
| Single-Hop | 841 / 810 / 440 / 500 / 686 | 841 / 811 / 440 / 500 / 687 |
