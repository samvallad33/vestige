# MemoryArena: preregistered protocol for the Vestige run

**Status: preregistered, not run. No Vestige number on MemoryArena exists.**

This file fixes the protocol before any data exists. When a number appears, it
was produced by exactly this protocol, or the deviation is written into the
Amendments section at the bottom before the number is quoted anywhere. The
preregistration is the commit that adds this file; the run must use a Vestige
build whose history contains that commit.

Harness: `benchmarks/memoryarena/`. Tracking issue: #242. Read
[`../BENCHMARKS.md`](../BENCHMARKS.md) first for why this page is written the
way it is: after the CauseBench retraction, a number a stranger cannot re-run is
worse than no number.

## 1. What MemoryArena measures, and why it is the first standard benchmark we fill in

MemoryArena (He et al., [arXiv:2602.16313](https://arxiv.org/abs/2602.16313),
code at <https://github.com/ZexueHe/MemoryArena>) evaluates memory inside a
Memory-Agent-Environment loop rather than as a retrieval quiz. An agent works
through a task made of dependent subtasks; before each subtask the memory
system wraps the prompt with whatever it chooses to surface, after each
subtask the agent's trajectory is stored, and later subtasks are only solvable
if what the agent learned earlier is retrieved and used. The paper's finding is
that agents near saturation on LoCoMo-style recall benchmarks score low here.
That gap, passive recall versus decision-relevant memory, is the thesis Vestige
is built on and the one our own Silent Rotation harness tests. MemoryArena is
the independent, third-party version of that question, which is why it goes
first.

The paper's published tables are not reproduced on this page. Read them from
the PDF. Nothing measured here is placed next to them (section 7).

## 2. Scope of the run

Two of the five task families, both from the released dataset
`ZexueHe/memoryarena` at the pinned revision:

| family (HF config) | tasks | subtasks per task | one task is |
| --- | --- | --- | --- |
| `formal_reasoning_math` | 40 | 2 to 16 | one paper: ordered questions, each with a background text; later questions build on earlier results |
| `formal_reasoning_phys` | 20 | 2 to 12 | same shape, physics papers |

Why these two: the environment is self-contained (no WebShop server, no travel
environment, no web-search API), the LLM judge is the only external call, and
the whole run reproduces from one machine with one API key. The other three
families (bundled shopping, group travel, progressive search) are out of scope
for this preregistration. Each gets its own preregistration before any run.

## 3. Pinned before the run

| what | value | where it is recorded |
| --- | --- | --- |
| MemoryArena code | `6cd9de14b71915e39ac742a20dc33785e14b6aab` (2026-06-01) | `benchmarks/memoryarena/MEMORYARENA.lock.json`; `install.py` refuses another HEAD without `--allow-unpinned` |
| Dataset | `ZexueHe/memoryarena` revision `da1a37c8b19280e18627ca01cf368195a5e1d92e`, CC-BY-4.0 | lock file |
| Evaluator | upstream `env/env_systems/formal_reasoning_env/eval.py` at the pinned revision, unmodified | run manifest |
| Task agent | `gpt-5-mini`, temperature 0.0, max_tokens 8192 (upstream config values, unchanged) | the config files in `benchmarks/memoryarena/configs/` |
| Judge | `gpt-5-mini`, temperature 1.0, max_tokens 4096 (upstream values, unchanged) | same |
| Loop | `max_steps` 10, `session_wise_memory` true, `judge_result_in_memory` false: memory stores the agent's own task, answer and tool calls, never the judge's verdict | same |
| Vestige | a `--release` build of a commit containing this file; `serverInfo.version`, the commit and the binary SHA-256 go in the manifest and the sidecar start record | `vestige-arena-log.jsonl` start record, `manifest.json` |
| Vestige configuration | defaults, plus: `forceCreate: true` on every write, `top_k` 3, `retrieval_mode` balanced, `detail_level` summary, one `scope` per task. Nothing else. The sidecar start record lists every `VESTIGE_*` environment variable name present so a ranking-related override cannot go unrecorded | adapter, sidecar |
| Embedding readiness | the first retrieval waits until `memory_status(view=health)` reports `embeddingReady`, or a probe recall returns a non-null `semanticScore`; the method and the wait are recorded. A run that never got there is keyword-only and is not this arm | sidecar |
| Data directory | a fresh `VESTIGE_ARENA_DATA_DIR` per family, so the store holds only what this run wrote | manifest |

`forceCreate` is the one non-default and it is there for comparability: the
unit the agent stored is the unit retrieval ranks, the same as upstream's BM25
and embedding arms, which index each chunk as written. Letting the ingest gate
merge or supersede entries would change the corpus the arms are compared on.

## 4. Arms

All five run on both families, same tasks, same task order, same agent, same
judge. None is optional. One variable changes between arms: what is put inside
`<memory_context>`.

| arm | `memory_system_name` | what it is | why it is here |
| --- | --- | --- | --- |
| `none` | `none` | nothing stored, nothing retrieved; the prompt shape is identical to every other arm | the floor. Anything it scores is the agent and the judge, not memory |
| `bm25` | `bm25` | upstream `RAGMemorySystem`, Okapi BM25 over the stored chunks, top_k 3 | the earned-complexity bar. A memory system that cannot beat this has not earned its complexity |
| `text-embedding-3-small` | `text-embedding-3-small` | upstream `RAGMemorySystem` with OpenAI embeddings, top_k 3 | the plain embedding-retrieval bar |
| `long_context` | `long_context` | upstream: the whole transcript in the prompt, up to 120k tokens | the paper's primary baseline |
| `vestige` | `vestige` | this adapter over a live `vestige-mcp` | the system under test |

The `none` arm is provided by our adapter because upstream's server at the
pinned revision has no such backend. It renders `<memory_context>\nNone\n
</memory_context>\nUser: <prompt>`, which is exactly what upstream's RAG arm
renders when it has nothing to return, so the agent sees the same shape.

The vestige arm renders retrieved memories as upstream's RAG arm does:
`<memory>...</memory>` blocks inside `<memory_context>`, then `User: <prompt>`.
The prompt differs between arms only in which memories were chosen.

Run order within a family: `none` first. If the floor is not near zero on PS
(section 6), stop and look at the judge before spending on the other arms. Then
`bm25`, `text-embedding-3-small`, `long_context`, `vestige`. All arms of a
family run inside the same 48 hours and the dates go in the manifest, because
the API models behind `gpt-5-mini` drift.

## 5. Metrics

Computed by upstream `eval.py`, unmodified, then read by
`benchmarks/memoryarena/analyze.py`, which was written before any data existed.

| metric | definition (upstream) | role |
| --- | --- | --- |
| PS, Progress Score | per task, correct subtasks / subtasks; then the mean over tasks (`avg_progress_score`) | primary |
| SR, Success Rate | fraction of tasks with every subtask correct (`overall_average_passrate`) | secondary, reported, never the basis of a claim |
| `cummulative_passrate_at_min_k` | pass rate by subtask position up to the shortest task, every task contributing | tertiary, reported as the curve: does the memory arm hold up as the dependency chain grows |

Confound metrics, from the vestige sidecar: mean `context_chars` per wrap,
hits per wrap, `semanticScore`-null rate, recall errors, mean stored-entry
bytes and the share past Vestige's 8192-byte embedding horizon (full-text
search still indexes the rest of such an entry).

Every aggregate is printed with its n beside it.

## 6. Decision rules

Applied by `analyze.py` without discretion.

1. **Reference arm is `bm25`.** Every other arm is paired against it task by
   task on PS. Wins, losses and ties are counted; ties are dropped; the exact
   two-sided binomial sign test gives p. Alpha is 0.05.
2. **"Vestige above BM25 on `<family>`" is claimed only if** p < 0.05, wins
   exceed losses, and mean PS(vestige) exceeds mean PS(bm25). Otherwise the
   result is written as "not separated at this n", and that sentence comes
   first, before any number, the way the MemConflict page does it.
3. **The same rule** gives the comparisons against `text-embedding-3-small` and
   `long_context`. They are reported; the BM25 comparison is the headline.
4. **Families are never pooled.** Math and physics are separate claims with
   separate n.
5. **SR never carries a claim.** At n of 40 and 20 with near-zero base rates,
   SR moves by one task.
6. **Floor check.** If PS(`none`) exceeds 0.15 on a family, the judge is
   awarding credit without memory on that family; that is reported and no
   claim is made on the family.
7. **Blob-size confound.** If the vestige arm's mean `context_chars` exceeds
   1.25 times three times the mean stored-entry length (the upstream RAG arms'
   budget: three chunks), every vestige claim on that family carries the
   confound in the same sentence.
8. **Readiness.** If the sidecar's readiness record says embeddings were not
   ready, the vestige arm was keyword-only. It is reported as such and its
   number is not quoted as Vestige's.
9. **One run per task per arm.** A task that crashed (API error, timeout) is
   rerun once and the rerun is logged. Nothing is rerun because of its result.
   Nothing is re-judged.

## 7. What will not be claimed

- **No placement next to the paper's tables.** The paper reports a
  GPT-5.1-mini backbone; the released configs run `gpt-5-mini`; the judge is
  stochastic at temperature 1.0; the API models drift between dates. Our
  numbers and the paper's do not share a table, a sentence, or a chart.
- **Nothing about shopping, travel or search.** Not run, not claimed.
- **No "state of the art", no ranking against systems we did not run** on the
  same day with the same agent and judge.
- **No significance where the sign test has nothing to work with.** p is
  reported with its paired n whatever the n is; a small n is printed, not
  hidden.
- **No cross-family pooling and no SR-based claim** (rules 4 and 5).

## 8. Run procedure

Placeholders in angle brackets. Keys go in the environment, never in a config
file, never in a commit.

```sh
# upstream, at the pinned revision
git clone https://github.com/ZexueHe/MemoryArena
cd MemoryArena
git checkout 6cd9de14b71915e39ac742a20dc33785e14b6aab
conda env create -f env/env_systems/formal_reasoning_env/environment.yml   # upstream's environment

# the adapter, the none arm and the run configs
python3 <vestige>/benchmarks/memoryarena/install.py --memoryarena .
# edit the copied configs: replace <OPENAI_BASE_URL> with your endpoint (no keys in files)

# Vestige, release build of a commit that contains this file
(cd <vestige> && cargo build --release -p vestige-mcp)
export VESTIGE_MCP_BINARY=<vestige>/target/release/vestige-mcp
export VESTIGE_ARENA_DATA_DIR=<fresh dir for this family>
export OPENAI_API_KEY=<key>

# servers (two terminals)
python env/env_server.py
python memory/server.py

# math, arms in the preregistered order; then eval
for cfg in math_none math_bm25 math_text_embedding math_longcontext_gpt-5-mini math_vestige; do
  python run_math.py -c configs/formal_reasoning_configs/$cfg.json
  python env/env_systems/formal_reasoning_env/eval.py configs/formal_reasoning_configs/$cfg.json
done
# physics: a fresh VESTIGE_ARENA_DATA_DIR, then
# phys_none phys_bm25 phys_text-embedding phys_longcontext_gpt-5-mini phys_vestige

# analysis, one call per family
python3 <vestige>/benchmarks/memoryarena/analyze.py \
  --arm none=results/json/math/none \
  --arm bm25=results/json/math/bm25 \
  --arm text-embedding-3-small=results/json/math/text-embedding-3-small \
  --arm long_context=results/json/math/long_context_gpt-5-mini \
  --arm vestige=results/json/math/vestige \
  --reference bm25 --sidecar $VESTIGE_ARENA_DATA_DIR/vestige-arena-log.jsonl \
  --out <vestige>/benchmarks/memoryarena/results/<UTC>/math.json
```

Before the run, the smoke test must pass against the exact binary that will be
used:

```sh
VESTIGE_MCP_BINARY=<vestige>/target/release/vestige-mcp python3 <vestige>/benchmarks/memoryarena/smoke_test.py
```

## 9. What gets checked in

`benchmarks/memoryarena/results/<UTC>/` containing:

- `manifest.json`: Vestige commit, binary SHA-256, `serverInfo`, MemoryArena
  and dataset revisions, agent and judge model names, API provider, start and
  end dates per arm, total API cost, the smoke test output.
- per family and arm: upstream's `all_results.json` and every
  `<paper>/result.jsonl` (the transcripts: memory-wrapped prompt, response,
  judge verdict), unedited.
- the vestige sidecar `vestige-arena-log.jsonl` for each family.
- `analyze.py` output JSON and the markdown tables it printed.

The README then gets one table row: both PS numbers with n, the sign test
verdict, and links to this file and the results directory. Not before.

## 10. Cost and time, an estimate

Upper bound on subtasks per arm: 40 tasks times 16 plus 20 tasks times 12, or
880; five arms give at most 4,400 agent calls and 4,400 judge calls, all on
`gpt-5-mini`. Upstream runs tasks sequentially, so wall-clock is dominated by
API latency: plan on a day for both families across five arms, and API cost
in the tens of dollars at list prices. These are estimates for planning, not
part of the protocol.

## 11. Amendments

None. Format for any future entry: date, what changed, why, and whether data
already existed when the change was made.
