# Benchmarks

This page documents the only retrieval benchmark Vestige currently stands
behind, what it does and does not measure, how to reproduce it from a clean
checkout, and where it is weak.

> **Retraction notice — CauseBench.**
> Vestige previously advertised a benchmark called **CauseBench**. It is
> **formally withdrawn**. Its harness no longer exists, its repro command
> 404'd, and its published numbers are retracted. **CauseBench must never be
> cited**, in release notes, READMEs, issues, launch posts, or conversation —
> not even as "an earlier result". A separate LongMemEval run was also
> invalidated and must not be cited. See `CHANGELOG.md`
> ("README rewrite; CauseBench replaced by Silent Rotation").
>
> That history is why this page is written the way it is. After a retraction,
> a number that cannot be reproduced by a stranger is worse than no number at
> all. Everything below is built so a stranger can re-run it and get the same
> answer, or catch us being wrong.

---

## MemConflict

**Paper:** [MemConflict: Evaluating Long-Term Memory Systems Under Memory
Conflicts](https://arxiv.org/abs/2605.20926) (arXiv:2605.20926)
**Upstream code and data:** <https://github.com/TaoZhen1110/MemConflict>
**Pinned revision:** `ec51d5d36e87f7665d1337f3a88cbde95fc2a964`
**Harness:** `benchmarks/memconflict/`

MemConflict measures whether a long-term memory system retrieves and ranks the
memory that is *temporally valid, factually correct, and contextually
applicable* when the store also contains conflicting alternatives. It defines
three conflict types:

| Conflict type | Validity dimension | Core challenge |
| --- | --- | --- |
| Dynamic | Temporal validity | Identify the current state after real updates. |
| Static | Factual correctness | Hold a true fact against a later false contradiction. |
| Conditional | Contextual applicability | Recover the right condition-value binding. |

This is the benchmark where Vestige *should* be strongest, because it ships
contradiction inspection as a first-class tool (`recall(mode="contradictions")`)
rather than as a post-hoc heuristic.

### Published numbers, and one correction

We verified the upstream numbers against the paper directly rather than
trusting a secondhand summary. One widely-repeated framing is wrong, so state
it precisely:

**Table 3, Average Answer Accuracy (AA)** — this is what the commonly-quoted
six-number list actually is:

| System | Average AA |
| --- | --- |
| MemOS | 0.5539 |
| Letta | 0.4871 |
| A-Mem | 0.4452 |
| Mem0 | 0.3612 |
| Memobase | 0.3553 |
| LangMem | 0.2822 |

**Table 3, Conflict Recognition Score (CRS)** — a *different* column, defined
only for static conflicts:

| System | Static CRS |
| --- | --- |
| A-Mem | **0.2501** |
| MemOS | 0.2361 |
| LangMem | 0.2083 |
| Letta | 0.2031 |
| Mem0 | 0.1528 |
| Memobase | 0.0694 |

Both tables above were read directly out of the paper PDF (Table 3, headed
"Dynamic AA / UOCS, Static AA / CRS, Conditional AA, Average AA"), not from a
summary. The paper states it plainly: "CRS remains low for all systems, with
the best score only reaching 0.2501" (MemConflict, arXiv:2605.20926, S4.3).

The `0.5539 / 0.4871 / ...` figures are **Answer Accuracy, not Conflict
Recognition Score**. Calling them CRS overstates the field by roughly a factor
of two. The best CRS achieved by *any* of the six systems is **0.2501**
(A-Mem). The entire field sits between 0.07 and 0.25 on actually recognising
contradictions — that is the real headroom, and it is much larger than the AA
column suggests.

### What our harness measures

For each simulated user, sessions are ingested in chronological order. After
each session, that session's questions are asked. A question therefore only
ever sees memories from sessions at or before it — no lookahead.

Four arms answer every question. **All four run every time. None is optional.**

| Arm | What it does | Why it exists |
| --- | --- | --- |
| `nomem` | No retrieval; reader gets an empty string. | The true floor. Anything scoring above ~0 here is measuring the judge, not memory. |
| `random` | K memories drawn uniformly at random (fixed seed) from the same corpus. | **Blob-inflation control.** The judge gives partial credit for token overlap, so *any* K memories score above zero by chance. An arm that cannot beat `random` is reporting corpus statistics. |
| `bm25` | Okapi BM25 (k1=1.5, b=0.75) over the identical corpus. | **The earned-complexity bar.** |
| `vestige` | `recall(...)` against a live `vestige-mcp` server. | The system under test. |

The BM25 and no-memory controls are non-negotiable because
[MemDelta](https://arxiv.org/abs/2606.29914) (arXiv:2606.29914) showed that
agent memory systems routinely fail against controlled baselines — agent
self-memory scored 42% where basic retrieval scored 47% on LongMemEval-S — and
that changing a single component (the embedding model alone shifted accuracy
by 6.2pp, p=0.004) can reverse system rankings. A memory system that cannot
beat naive BM25 on the same corpus, with the same reader and the same judge,
has not earned its complexity. The harness prints that verdict explicitly.

**Metrics** (ported faithfully from upstream `Evaluation/eval_scoring.py`):

- `answer_accuracy` (AA) — partial-credit match against the gold answer.
- `UOCS` — dynamic conflicts: did the output show update *and* ordering?
- `CRS-lex` — static conflicts: upstream's exact keyword rule
  (`inconsisten|conflict|contradict|cannot confirm|uncertain|mismatch`),
  applied identically to every arm. Apples to apples.
- `CRS-struct` — **Vestige only.** Did `recall(mode="contradictions")` return
  at least one contradiction pair? Scored *structurally* from
  `contradictionsFound`, never by keyword-matching the response, so the metric
  cannot be satisfied by the server merely emitting the word "contradiction".

Type-level means are macro-averaged, matching the paper's "Average AA" column.
Metrics that do not apply to a conflict type are **omitted, never scored zero**,
so denominators stay correct.

### What this does NOT measure

Read this section before quoting any number from this harness.

1. **These numbers are not comparable to the paper's Table 3.** Upstream's
   headline results used an **LLM judge (gpt-5.0-mini)** and an LLM reader. We
   use upstream's deterministic **rule-based fallback judge** and a **non-LLM
   reader**. Absolute values are on a different scale. Never place our AA next
   to a published AA in the same table or sentence.
2. **Only cross-arm differences within a single run are meaningful.** The arms
   share one corpus, one reader, one judge, one K. That comparison is valid.
   Nothing else is.
3. **The reader is not an agent.** It concatenates the top-K retrieved memory
   texts. It does not reason, disambiguate, or resolve conflicts. So AA here is
   a *retrieval-quality proxy* — "was the evidence surfaced" — not end-to-end
   task accuracy. This is deliberate: it removes the model confound that
   MemDelta identifies as the field's main source of bogus results.
4. **CRS-struct is not a head-to-head win.** BM25 and the random/no-memory
   controls have no contradiction channel at all, so they are structurally 0
   by construction, not by measured deficit. CRS-struct reports whether a
   Vestige *capability* fires. It is not evidence that Vestige beats BM25 at
   contradiction recognition, and must never be presented as such.
5. **White-box metrics are not implemented.** The paper's SEH@K and SRS require
   gold supporting-memory IDs. The released `Data/Step4_4.jsonl` questions carry
   only `question / answer / conflict_type / ability_target / difficulty` — no
   gold memory ID — so SEH@K and SRS **cannot be computed from the public data**
   and are not reported. We will not estimate them.
6. **Released data ≠ evaluated data.** The paper reports ~12 virtual users;
   the released repo ships an *expanded* release of 30 instances. Our subset is
   drawn from the 30. Our instance sample is therefore not the paper's sample.
7. **Subset, not full benchmark.** Runs are capped by `--instances` and
   `--sessions`. A capped run is a real measurement of a smaller slice, not an
   estimate of the full 3,750-question benchmark. The cap is recorded in the
   results JSON. Small subsets have wide error bars; we report no significance
   tests and you should not infer any.
8. **No statistical significance testing.** With a handful of static-conflict
   questions per run, differences of a few points are noise. Do not read them
   as signal.
9. **Ingestion is user-turn only.** Assistant turns are excluded as agent
   paraphrase rather than facts about the user. That is a modelling choice and
   it changes the corpus every arm sees.

### Reproducing

Requires: Rust toolchain, Python 3.9+ (**standard library only — no pip
install**), network access for the one-time dataset fetch.

```sh
# 1. Build the server
cargo build -p vestige-mcp

# 2. Fetch the pinned dataset (verifies SHA-256; refuses to run on a mismatch)
python3 benchmarks/memconflict/fetch_dataset.py

# 3. Run all four arms
python3 benchmarks/memconflict/run.py --instances 2 --sessions 25 --top-k 5
```

The dataset is **not vendored**. It is downloaded at the pinned commit and
hash-verified; a mismatch is a hard failure, never a warning.

Every run writes `benchmarks/memconflict/results/memconflict-<UTC>.json`
containing the exact command, the dataset revision and file hashes, the Vestige
git commit / branch / dirty flag, the MCP `serverInfo`, full config (arms,
top-k, recall mode, seed, reader, judge), the observed embedding-warmup record,
machine specs, per-question records, and the caveat list. The console prints
the exact reproduction command on exit.

**Determinism.** Given the same dataset revision, Vestige commit, and flags,
the `nomem`, `random` (seeded) and `bm25` arms are fully deterministic. The
`vestige` arm is *not* guaranteed bit-identical across runs: FSRS retention
state, timestamps, and consolidation are time-dependent by design. Re-running
reproduces the ranking, not necessarily the fourth decimal place.

**Embedding warmup.** The harness blocks for a mandatory warmup (default 45s)
after `initialize` before the first `tools/call`. Skipping it silently measures
a degraded keyword-only fallback — a failure that does not look like a failure.
The observed warmup is recorded in every results file. Verify a run used real
embeddings by checking that `semanticScore` is non-null in retrieval output.

### Known limitations of the harness itself

- Per-user isolation uses Vestige's `scope` namespace, not a fresh database per
  user. This is a speed tradeoff: a fresh data dir per user would pay the
  warmup cost repeatedly. Scope isolation is enforced by the server
  (`includeCrossScope` defaults false), but it is a weaker guarantee than
  process isolation.
- `smart_ingest` batches at 20 items per call (the tool's cap). Ingest
  throughput measured on the reference machine was ~7 units/s, which is what
  bounds run size.
- The rule-based judge is lexical. It rewards token overlap and cannot detect a
  semantically correct paraphrase that shares no vocabulary with the gold
  answer. It under-credits every arm, but it under-credits them *equally*,
  which is what keeps the cross-arm comparison sound.

---

## First run: what the instrument actually reported

Recorded so the instrument is judged on what it produced, not on what we hoped
it would produce.

```
python3 benchmarks/memconflict/run.py --instances 2 --sessions 25 --top-k 5
```

2 simulated users, 25 sessions each, 2,207 memory units ingested, 98 questions,
top-k 5, Vestige 2.4.1, Apple M1 Max. Full record:
`benchmarks/memconflict/results/memconflict-20260830T173831Z.json`.

| arm | macroAA | microAA | dynAA | UOCS | statAA | CRS-lex | condAA | chars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nomem | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| random | 0.0322 | 0.0867 | 0.0966 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 687 |
| bm25 | **0.4881** | 0.2704 | 0.2500 | 0.1932 | 0.2143 | 0.0000 | 1.0000 | 736 |
| vestige | 0.4416 | **0.3163** | **0.3011** | **0.3636** | **0.3571** | 0.0000 | 0.6667 | 854 |

n: dynamic 88, static 7, conditional 3.

**The controls behaved.** `nomem` scored a clean 0.0000 on everything, so the
judge awards nothing for an empty answer. `random` scored 0.0322 macro, so
blob-inflation is real but small. Both floors are where they should be, which is
what licenses reading anything else on this table.

**Vestige does not beat BM25 on the headline metric.** macro AA 0.4416 vs
0.4881. Reported first because it is the metric the paper leads with and it is
the one where Vestige loses.

**But macro and micro disagree, and the reason is a 3-question subset.**
`macroAA` weights conditional conflicts (n=3) the same as dynamic (n=88). BM25
went 3/3 there and Vestige 2/3; that single question is the entire macro gap.
On micro AA, Vestige leads 0.3163 to 0.2704, and it leads on dynamic AA, UOCS
and static AA. **The honest statement is that this run does not separate them.**
n=3 cannot support a ranking either way, and we ran no significance test.

**The blob-size caveat on Vestige's micro-AA lead.** Vestige hands the judge 854
chars per question against BM25's 736 (1.16x). Vestige merges near-duplicate
units into composite nodes (marked `[MERGED]`, up to 719 chars) even under
`batchMergePolicy=force_create`, so its retrieval units are larger. Content is
not lost, but a token-overlap judge rewards more text. Vestige's micro-AA edge
points the same direction as its larger blob, so that edge is **not clean**. The
harness now records `reader_chars_mean` on every run and prints a CONFOUND
WARNING past 1.25x; at 1.16x it did not fire, but the effect is not zero.

### The finding that matters: contradiction recognition did not fire

**CRS-struct = 0.0000. CRS-lex = 0.0000 for every arm.**

On the benchmark built to test contradiction handling, on the capability Vestige
ships as a first-class tool, `recall(mode="contradictions")` returned **zero**
contradiction pairs for all 7 static-conflict questions.

This is not a harness artifact. We checked:

- Raising `limit` from the default 50 to the maximum 200: still 0.
- Short keyword topics (`university`, `gender`, `residence`) instead of full
  question text: still 0.
- No topic at all (scan recent memories), limit 200: still 0.
- **The contradicting evidence is present and retrievable.** A `lookup` recall
  returns both "I studied at MIT" and the Cal State Long Beach memory, and both
  "I have a Bachelor degree" and "I have a Master of Science", from the same
  store, in the same scope. The pairs are there. The detector does not fire on
  them.

For scale, the published field ceiling on this metric is 0.2501 (A-Mem), and the
paper's own summary is that CRS is low for every system it tested. Vestige is
currently at 0.0000 against that ceiling on this subset.

Two engine-side issues surfaced while establishing this, neither of which we
changed (out of scope for this harness work):

1. **`recall(mode="contradictions")` accepts no `scope` parameter.** It calls
   `hybrid_search` / `get_all_nodes` without namespace filtering, so it reads
   across every project scope in the database. In this harness that means the
   probe was not isolated per simulated user. It is also a correctness concern
   outside benchmarking.
2. **Detection is a strict lexical heuristic** (`topic_overlap >= 0.4` plus
   `appears_contradictory`), which is a plausible reason it misses
   contradictions phrased in natural conversational language, which is exactly
   what MemConflict plants.

Neither is diagnosed further here. The harness's job was to surface them
measurably, and it did.

## LongMemEval_S — sanity check only

**Harness:** `benchmarks/memconflict/longmemeval.py`
**Dataset:** `xiaowu0162/longmemeval-cleaned` (MIT, ungated)
**Pinned revision:** `98d7416c24c778c2fee6e6f3006e7a073259d48f`
**Status:** wired and runnable. **Not a headline benchmark. Never quote these
numbers as a LongMemEval score.**

A previous Vestige LongMemEval run was **invalidated** (absolute-value scoring
bug, 8192-byte truncation, no-op weights) and **must not be cited**.

While pinning this we found a second, independent reason that any older Vestige
LongMemEval number is unusable: **the original `xiaowu0162/longmemeval` dataset
is deprecated upstream.** It is replaced by `longmemeval-cleaned`, which removes
noisy history sessions that interfered with answer correctness. A result
computed against the deprecated original is invalid on that ground alone,
regardless of the scoring bugs. This harness pins the cleaned release.

**What it measures.** One metric: `evidence_recall@k` — after ingesting a
question's haystack, does the concatenation of the top-k retrieved memories
contain the gold answer string (normalised)? It is deliberately
reader-independent and unambiguous, and asks exactly one thing: *did retrieval
put the answer in front of the reader?*

**What it does not measure.** Anything else. There is no reader, no judge, and
no end-to-end task score. A real LongMemEval result needs both and would be a
much larger claim. This exists to catch a silently broken harness — a retriever
returning nothing, an ingest path dropping content, an embedding service that
never warmed up — on a dataset entirely independent of MemConflict.

```sh
python3 benchmarks/memconflict/longmemeval.py --questions 5
```

The same four arms run here as in MemConflict. Default is 5 of 500 questions;
each question carries a ~50-session haystack, so full runs are expensive and
bounded by ingest throughput.

---

## Rules for citing any number from this page

1. Always cite the arm, the dataset revision, the instance/session cap, and the
   judge. A bare number is not a result.
2. Never compare our absolute values to a published table.
3. Always report the `bm25` control next to any `vestige` number. If Vestige
   did not beat BM25, say so.
4. Never cite CauseBench. Never cite the invalidated LongMemEval run.
5. If a number cannot be reproduced from a clean checkout with the printed
   command, it is not a number — retract it.
