# Scoreboard — pool A

Baseline is the combined Step 31 + 32 + 33 + 34 tip, measured once before any new seat. Its tree is the algo-44 tree `4612c06f` (the published bd2693e2 board) plus Steps 31, 32, 33 and 34. Algo-44 versus the tip: 0 per-question coordinate losses on LME, LoCoMo and BEAM (3 / 12 / 4 coordinate gains). Every listed Step 31–34 flip is still a Turn@1 hit on the tip.

After is the release binary with three seats, in order: the buried lexical peak, the anaphoric tail, then the unseated session peak.

## Headline

| Bench | Metric | Tip 31+32+33+34 | After three seats |
| --- | --- | ---: | ---: |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 353/428/437 | 353/428/437 |
| LME_S | Turn All@5 | 314 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 808/1127/1226 | 808/1134/1229 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 160/238/267 | 160/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 | 118 / 138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 |

LoCoMo Turn@5 1127→1134 and Turn@10 1226→1229. Every other headline cell is unchanged. Per-question losses versus the tip: 0 on every metric of all three benches. LME Session drops: 0.

## LoCoMo by category

| Category | n | Tip T@1/@5/@10 | After T@1/@5/@10 | Tip S@1/@5 | After S@1/@5 |
| --- | ---: | --- | --- | --- | --- |
| Multi-Hop | 282 | 97/158/186 | 97/158/186 | 156/246 | 156/246 |
| Temporal | 321 | 191/248/260 | 191/250/262 | 242/299 | 242/299 |
| Open-Domain | 92 | 20/35/43 | 20/35/43 | 36/65 | 36/65 |
| Single-Hop | 841 | 500/686/737 | 500/691/738 | 699/806 | 699/806 |

## Questions that moved

| Question | Category | Seat | Tip | After |
| --- | --- | --- | --- | --- |
| `9:137` | Single-Hop | buried lexical peak | `[1, 1, 0, 0, 0]` | `[1, 1, 0, 1, 1]` |
| `7:146` | Single-Hop | anaphoric tail | `[0, 0, 0, 0, 1]` | `[0, 0, 0, 1, 1]` |
| `9:113` | Single-Hop | anaphoric tail | `[0, 1, 0, 0, 1]` | `[0, 1, 0, 1, 1]` |
| `9:131` | Single-Hop | anaphoric tail | `[0, 1, 0, 0, 1]` | `[0, 1, 0, 1, 1]` |
| `9:146` | Single-Hop | anaphoric tail | `[1, 1, 0, 0, 1]` | `[1, 1, 0, 1, 1]` |
| `1:10` | Temporal | unseated session peak | `[1, 1, 0, 0, 0]` | `[1, 1, 0, 1, 1]` |
| `6:64` | Temporal | unseated session peak | `[1, 1, 0, 0, 0]` | `[1, 1, 0, 1, 1]` |

`9:137` "When did Calvin first get interested in cars?", evidence `D26:6`. Unique named-speaker declarative whose `t_own` beats the scene pool of 50. It takes the lowest-own question seat in slots 1–4. Turn@1 stays 0 because slot 0 does not move.

The anaphoric-tail rows were already Turn@10 hits. The tail maximum is an anaphoric declarative whose own score beats every top-5 turn, and it swaps with the lowest-own question in slots 1–4.

- `7:146` "What kind of yoga routine does Deborah recommend to Jolene?", evidence `D16:15`.
- `9:113` "What activity did Calvin enjoy during his summer drives?", evidence `D20:4`.
- `9:131` "What inspired Calvin's recent music?", evidence `D25:10`.
- `9:146` "What type of music has Dave been getting into lately?", evidence `D28:40`.

The unseated-session rows were already Session@1 hits and already had a gold inside pool@50, outside the returned top 10. The unique declarative outside the returned list whose own score beats every returned turn, from a session with no top-5 turn, takes the lowest-own question seat.

- `1:10` "When did Gina team up with a local artist for some cool designs?", evidence `D5:5`.
- `6:64` "How long did John practice chess for before winning the chess tournament?", evidence `D17:1`, `D30:2`, `D30:4`. `D30:2` and `D30:4` were already in the pool.

## Gates

`cargo clippy -p vestige-core --all-targets -- -D warnings` clean on this seat. `cargo check -p vestige-mcp` clean. `cargo test -p vestige-core` 566 passed. Hash-check 0/500 on ranked ids and 0/500 on full responses.

## Stage funnel

Columns are gold in corpus / pool@50 / scene-rank 0 / final rank 0 / final top 5. Measured with `tools/c2_funnel.py`.

| Category | Tip | After buried peak | After anaphoric tail | After unseated session peak |
| --- | --- | --- | --- | --- |
| Multi-Hop | 282 / 243 / 72 / 97 / 158 | 282 / 243 / 72 / 97 / 158 | 282 / 243 / 72 / 97 / 158 | 282 / 243 / 72 / 97 / 158 |
| Temporal | 320 / 299 / 182 / 191 / 248 | 320 / 299 / 182 / 191 / 248 | 320 / 299 / 182 / 191 / 248 | 320 / 299 / 182 / 191 / 250 |
| Open-Domain | 92 / 63 / 14 / 20 / 35 | 92 / 63 / 14 / 20 / 35 | 92 / 63 / 14 / 20 / 35 | 92 / 63 / 14 / 20 / 35 |
| Single-Hop | 841 / 810 / 440 / 500 / 686 | 841 / 811 / 440 / 500 / 687 | 841 / 811 / 440 / 500 / 691 | 841 / 811 / 440 / 500 / 691 |
