# Scoreboard — pool A

Baseline is the combined Step 31 + 32 + 33 + 34 tip, measured once before either seat. Its tree is the algo-44 tree `4612c06f` (the published bd2693e2 board) plus Steps 31, 32, 33 and 34. Algo-44 versus the tip: 0 per-question coordinate losses on LME, LoCoMo and BEAM (3 / 12 / 4 coordinate gains). Every listed Step 31–34 flip is still a Turn@1 hit on the tip.

After is the release binary with both seats: the buried lexical peak, then the anaphoric tail.

## Headline

| Bench | Metric | Tip 31+32+33+34 | After both seats |
| --- | --- | ---: | ---: |
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 353/428/437 | 353/428/437 |
| LME_S | Turn All@5 | 314 | 314 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 808/1127/1226 | 808/1132/1227 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 160/238/267 | 160/238/267 |
| BEAM-100K | Turn All@5 / All@10 | 118 / 138 | 118 / 138 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 |

LoCoMo Turn@5 1127→1132 and Turn@10 1226→1227. Every other headline cell is unchanged. Per-question losses versus the tip: 0 on every metric of all three benches. LME Session drops: 0. The same holds versus the buried-peak commit alone: the anaphoric tail adds four Turn@5 hits and no coordinate losses.

## LoCoMo by category

| Category | n | Tip T@1/@5/@10 | After T@1/@5/@10 | Tip S@1/@5 | After S@1/@5 |
| --- | ---: | --- | --- | --- | --- |
| Multi-Hop | 282 | 97/158/186 | 97/158/186 | 156/246 | 156/246 |
| Temporal | 321 | 191/248/260 | 191/248/260 | 242/299 | 242/299 |
| Open-Domain | 92 | 20/35/43 | 20/35/43 | 36/65 | 36/65 |
| Single-Hop | 841 | 500/686/737 | 500/691/738 | 699/806 | 699/806 |

## Questions that moved

| Question | Category | What | Tip | After |
| --- | --- | --- | --- | --- |
| `9:137` | Single-Hop | buried lexical peak | `[1, 1, 0, 0, 0]` | `[1, 1, 0, 1, 1]` |
| `7:146` | Single-Hop | anaphoric tail | `[0, 0, 0, 0, 1]` | `[0, 0, 0, 1, 1]` |
| `9:113` | Single-Hop | anaphoric tail | `[0, 1, 0, 0, 1]` | `[0, 1, 0, 1, 1]` |
| `9:131` | Single-Hop | anaphoric tail | `[0, 1, 0, 0, 1]` | `[0, 1, 0, 1, 1]` |
| `9:146` | Single-Hop | anaphoric tail | `[1, 1, 0, 0, 1]` | `[1, 1, 0, 1, 1]` |

`9:137` "When did Calvin first get interested in cars?", evidence `D26:6`. That turn is the unique named-speaker declarative whose `t_own` beats the scene pool of 50. It takes the lowest-own question seat in slots 1–4. Slot 0 stays, so Turn@1 does not move. Turn@10 moves because the gold was outside the returned list.

The other four were already Turn@10 hits. Each gold is an anaphoric declarative sitting in ranks 5–9 whose own score beats every top-5 turn. It swaps with the lowest-own question in slots 1–4. The top-10 set does not change, so Turn@10 stays 1 and Turn@5 becomes 1.

- `7:146` "What kind of yoga routine does Deborah recommend to Jolene?", evidence `D16:15`.
- `9:113` "What activity did Calvin enjoy during his summer drives?", evidence `D20:4`.
- `9:131` "What inspired Calvin's recent music?", evidence `D25:10`.
- `9:146` "What type of music has Dave been getting into lately?", evidence `D28:40`.

## Gates

`cargo clippy -p vestige-core --all-targets -- -D warnings` clean on this seat. `cargo check -p vestige-mcp` clean. `cargo test -p vestige-core` 565 passed. Hash-check 0/500 on ranked ids and 0/500 on full responses.

## Stage funnel

Columns are gold in corpus / pool@50 / scene-rank 0 / final rank 0 / final top 5. All three columns were measured with `tools/c2_funnel.py`. The anaphoric tail swaps two turns already in the seated prefix, and the measured pool, scene-rank 0, and final rank 0 stay on the buried-peak numbers. Single-Hop final top 5 moves 687→691.

| Category | Tip | After buried peak | After anaphoric tail |
| --- | --- | --- | --- |
| Multi-Hop | 282 / 243 / 72 / 97 / 158 | 282 / 243 / 72 / 97 / 158 | 282 / 243 / 72 / 97 / 158 |
| Temporal | 320 / 299 / 182 / 191 / 248 | 320 / 299 / 182 / 191 / 248 | 320 / 299 / 182 / 191 / 248 |
| Open-Domain | 92 / 63 / 14 / 20 / 35 | 92 / 63 / 14 / 20 / 35 | 92 / 63 / 14 / 20 / 35 |
| Single-Hop | 841 / 810 / 440 / 500 / 686 | 841 / 811 / 440 / 500 / 687 | 841 / 811 / 440 / 500 / 691 |
