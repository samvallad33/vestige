# SCOREBOARD Step 40 — Cue validity

Baseline is the combined Steps 31–38 plus Pool C, Pool B, and Pool A tip, replayed with `git am -3` from `d121d063` onto `perf/c2-step40-cue-validity`. Replayed tip before this step: `9d95cc42a6778cc5e044c6a244c2fe12c714c915` (pack commit `a2a7f2b92d03fb4a738fa300607f57837a071e7c` is the `#[test]` fix; hook order matches the pack README). Baseline `retrieval_bench` sha256 `dd449987923b653de2423edb495553da47d9ee005a5a5d2ea6b875be4ff1afbf`. After sha256 `59459169f862745acb45b88459c604fb41feba8b9ed5fcbd772ffcf6ba38c350`.

The only per-question change is LoCoMo `4:74` Turn@1 0→1. Every other aggregate below is unchanged. Zero losses.

| bench | metric | before | after |
|---|---|---:|---:|
| LME_S n=470 | Session @1/@5/@10 | 430/463/468 | 430/463/468 |
| LME_S | Session All@5 / All@10 | 424 / 447 | 424 / 447 |
| LME_S | Turn @1/@5/@10 | 355/429/438 | 355/429/438 |
| LME_S | Turn All@5 | 320 | 320 |
| LoCoMo n=1536 | Turn @1/@5/@10 | 812/1134/1229 | **813**/1134/1229 |
| LoCoMo | Session @1/@5 | 1133/1416 | 1133/1416 |
| LoCoMo | Multi-Hop Turn@1 / Session@1 | 97 / 156 | 97 / 156 |
| LoCoMo | Temporal Turn@1 / Session@1 | 192 / 242 | 192 / 242 |
| LoCoMo | Open-Domain Turn@1 / Session@1 | 20 / 36 | 20 / 36 |
| LoCoMo | Single-Hop Turn@1 / Session@1 | 503 / 699 | **504** / 699 |
| BEAM-100K n=355 | Turn @1/@5/@10 | 162/240/269 | 162/240/269 |
| BEAM-100K | Turn All@5 / All@10 | 122 / 142 | 122 / 142 |
| BEAM-100K | Session @1/@5/@10 | 133/249/275 | 133/249/275 |
| BEAM-100K | Session All@5 / All@10 | 137 / 178 | 137 / 178 |

LME Turn@1 by type, unchanged: single-session-user 55/64, multi-session 90/121, single-session-preference 12/30, temporal-reasoning 94/127, knowledge-update 62/72, single-session-assistant 42/56.

Per-question: LoCoMo `4:74` Turn@1 0→1. LME 0 diffs / 470. BEAM 0 diffs / 355. See `STEP40_PERQ_DIFF.md`.
