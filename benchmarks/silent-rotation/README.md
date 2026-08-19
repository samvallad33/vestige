# Silent Rotation

A benchmark that isolates the memory layer of a multi-agent coding fleet.

Three agents fix one failing end-to-end test in a TypeScript monorepo. The fix
requires the currently live signing key id. That key is randomized per trial from
a 50-key keyring, appears in no file the agents can read, and exists only in the
memory layer. Arms: a no-memory control, dense cosine RAG, Vestige, mem0,
SuperMemory, Hindsight, and Zep/Graphiti.

I built this benchmark and I also built one of the arms in it. That is why the
raw agent transcripts ship here instead of a summary table. I ran the same task
on two models. They are two experiments, not one pooled result. Limitations and
unequal tool sets are in [`FAIRNESS.md`](FAIRNESS.md). Reprint both tables with
`python3 tests/by_model_tables.py`.

## GPT-5.6 Sol — no memory 0/5, dense RAG 0/5, Vestige 5/5

![GPT-5.6 Sol — three arms, five trials](figures/silent-rotation-gpt-5.6-sol.png)

Five trials, three memory conditions, 45 transcripts, all on disk. Full write-up:
[`GPT-5.6-SOL.md`](GPT-5.6-SOL.md).

## Kimi K3 — SuperMemory is also 5/5

![Kimi K3 — transcript-backed](figures/silent-rotation-kimi-k3.png)

On this model SuperMemory scored 5/5. Vestige scored 4/4 on the cells whose
transcripts are on disk. What still separates Vestige is the first memory call:
the live key was already in that payload 12/12 times. Full write-up:
[`KIMI-K3.md`](KIMI-K3.md). The trial-1 deep dive, including the two-right
one-wrong merges, is [`FINDING.md`](FINDING.md).

## Ablation — three axes, 0/5 each

On GLM-5.2, dropping event-anchoring, the causal edge, or the shared
coordination bus each scored 0/5 against a 5/5 full Vestige run on the same
keys. No single axis reproduces the result. Write-up:
[`ABLATION.md`](ABLATION.md).


## Pre-registration

The ablation was designed to test whether the result survives without Vestige.
The arms, the thresholds, and the interpretation of every outcome — including
the ones that demote my own system — are committed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) **before the experiment runs**.

Check `git log` on that file against the results commit. If the interpretation
was chosen after the numbers came back, the commit order will show it.


---

## Reproduce the central claim in two seconds

No API keys. No network. No ollama. No docker. Python standard library only.

```sh
python3 tests/bm25_baseline.py results/runA-trial-1/corpus-export.json --no-dense
```

You should see:

```
  seeded memories in the store : 9
  retrievable competitor corpus: 8   (outage row excluded, as runner.py does)
  causal memories present      : 1
  key identifiers in corpus    : ['k_antares', 'k_nashira', 'k_vela']

  rank of the causal memory out of 8, lower is better
  query (verbatim from a transcript)  dense    bm25
  ----------------------------------------------------
  rag a0, first query                 -        #7
  zep a0, first query                 -        #7
  zep a0, third query                 -        #7
  mem0 a0, second query               -        #4
  supermemory a0, first               -        #6
```

That is the whole finding, on your machine, in one command. The queries are
verbatim from the agent transcripts in `results/`. They are not authored for the
script. The memory that explains the failure ranks 7th of 8 under BM25, and the
decoy ranks 1st.

Drop `--no-dense` to also run dense cosine. That path needs ollama on
`localhost:11434` with `nomic-embed-text`, the same embedder the `rag` arm uses.
Dense buries the causal memory at the same rank.

**Note:** the input must be `corpus-export.json` (the memory-layer export).
`prod-corpus.json` in the same directories is a different artifact, the
production replay oracle, and is not a corpus.

---

## What is in here

| Path | What it is |
|---|---|
| `FINDING.md` | Trial 1 on Kimi K3, seven backends. Start here for the argument. |
| `GPT-5.6-SOL.md` | GPT-5.6 Sol: no memory 0/5, dense RAG 0/5, Vestige 5/5. 45 transcripts. |
| `KIMI-K3.md` | Kimi K3: seven backends. SuperMemory 5/5. Vestige 4/4 with transcripts on disk. |
| `FAIRNESS.md` | Tool sets, planted causal edge, missing transcripts, withheld cells. |
| `ABLATION.md` | GLM-5.2: three axes, 0/5 each. |
| `figures/silent-rotation-gpt-5.6-sol.png` | GPT-5.6 Sol, three arms. |
| `figures/silent-rotation-kimi-k3.png` | Kimi K3, transcript-backed. |
| `figures/silent-rotation-seven-backends.png` | Pooled across models. Not one experiment. Numbers from `EVIDENCE.md`. |
| `EVIDENCE.md` | Every claim in `FINDING.md` traced to a file and line. |
| `QUOTES.md` | Verbatim agent reasoning, including the passages where an agent reads the decoy's caveat and uses the key anyway. |
| `results/` | 25 trial folders across 6 models. Most cells have per-arm JSON plus one transcript per agent. Three cells have JSON scores and no transcripts (`runB-trial-1` anarchy/rag/sync); two more anarchy-only folders have JSON and no traces. `FAIRNESS.md` lists them. |
| `results/WITHHELD-contaminated/` | Cells excluded from the published scores, with a README explaining exactly what my harness did wrong in each. |
| `harness/` | The full runner. See "running the whole thing" below. |
| `tests/bm25_baseline.py` | The standalone lexical + dense baseline above. |
| `tests/by_model_tables.py` | Reprints the GPT and Kimi tables from `results/` JSON. No network. |
| `tests/arm_liveness.py`, `tests/verify-arms.sh` | Checks that every arm actually retrieves, rather than failing silently and logging a loss. |

---

## Reading a trial

```sh
ls results/runB-trial-3/
```

Each trial directory contains one `<arm>.json` per arm with the outcome, and one
`transcript-<arm>-a<N>.json` per agent with that agent's full reasoning and tool
calls. `manifest.json` records the trial's correct key and configuration.

The fastest way to see the failure mode is to read the three
`transcript-rag-a*.json` files in a trial the RAG arm lost, and watch three agents
independently arrive at different keys.

---

## Running the whole thing

Be aware of what this costs before you start. The full 7-arm sweep needs:

- an API key for whichever model provider you point it at, and real spend per
  trial (all seven arms run a three-agent fleet to completion)
- `ollama` running locally, for the dense-cosine embedder and for Graphiti's
  extraction model
- `docker`, for the FalkorDB instance Zep/Graphiti stores its graph in
- Node and a working `npm install` in the torture repo, since the fleet actually
  builds and tests it

```sh
cd harness
./run-experiment.sh
```

Individual arms are `run-<arm>.sh`. `aggregate-experiment.py` produces the
outcome tables.

If you only want to check the claim rather than reproduce the spend, the two
second command at the top of this file is the honest short path, and the
transcripts are the audit trail for everything else.

---

## Known limitations

These are stated at greater length in `FINDING.md`, and none of them are hidden.

- The causal edge is hand-authored. `harness/agent/prepare_trial.py` tags the
  cause and the failure with a shared entity. This measures traversal of an
  existing edge, not discovery of one.
- The retrievable corpus is 8 documents. That is easy mode for the RAG arms, and
  nothing here is measured at production scale.
- The three-axis ablation has been run. Event-anchoring, the causal edge, and
  the shared bus are load-bearing together (`ABLATION.md`). No single axis
  reproduces the full arm. The edge is still hand-authored: this measures
  traversal, not discovery.
- On Kimi K3, dense RAG is 3/4 correct (transcript-backed) and never agreed on a
  wrong key. On GPT-5.6 Sol the same RAG setup is 0/5. Those are different
  experiments. I do not pool them into one RAG score.
- I initially broke the mem0 and Zep arms by failing to flush state between
  trials, which disadvantaged them. The harnesses are fixed and those arms were
  re-run clean. Both the broken and the repaired cells are published.
