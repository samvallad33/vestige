# Silent Rotation

A benchmark that isolates the memory layer of a multi-agent coding fleet.

Three agents fix one failing end-to-end test in a TypeScript monorepo. The fix
requires the currently live signing key id. That key is randomized per trial from
a 50-key keyring, appears in no file the agents can read, and exists only in the
memory layer. Seven arms: a no-memory control, dense cosine RAG, Vestige, mem0,
supermemory, hindsight, and Zep/Graphiti.

I built this benchmark and I also built one of the arms in it. That is why all
246 raw agent transcripts ship here instead of a summary table.

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
| `FINDING.md` | The write-up. Start here if you want the argument. |
| `EVIDENCE.md` | Every claim in `FINDING.md` traced to a file and line. |
| `QUOTES.md` | Verbatim agent reasoning, including the passages where an agent reads the decoy's caveat and uses the key anyway. |
| `results/` | 25 trials across 6 models. Every trial has per-arm results and one transcript per agent. |
| `results/WITHHELD-contaminated/` | Cells excluded from the headline numbers, with a README explaining exactly what my harness did wrong in each. |
| `harness/` | The full runner. See "running the whole thing" below. |
| `tests/bm25_baseline.py` | The standalone lexical + dense baseline above. |
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
- No ablation separates event anchoring from causal traversal from temporal
  validity. You can fairly argue the result comes from removing query variance
  alone. That is the next experiment.
- On Kimi K3 the dense cosine baseline never converged wrong (3 correct, 2 split of 5) and it
  costs less per run. Across all 23 shared trials it is 4 correct to Vestige's 20, so the gap
  is model-dependent rather than uniform.
  is no "beats RAG on outcomes" claim here to defend.
- I initially broke the mem0 and Zep arms by failing to flush state between
  trials, which disadvantaged them. The harnesses are fixed and those arms were
  re-run clean. Both the broken and the repaired cells are published.
