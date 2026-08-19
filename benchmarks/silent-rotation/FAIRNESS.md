# Fairness

I built this benchmark and I also built one of the arms. That is a conflict of
interest. This page is the audit: what a skeptical reader should check, what
the files show, and what is still unequal.

GPT-5.6 Sol and Kimi K3 are separate experiments. They do not retrieve the
same way. Pooling them into one table hides that Kimi dense RAG is 3/4 while
GPT dense RAG is 0/5. The two runs are [`GPT-5.6-SOL.md`](GPT-5.6-SOL.md) and
[`KIMI-K3.md`](KIMI-K3.md).

Reprint the split from the JSON:

```sh
python3 tests/by_model_tables.py
```

## Reasonable objections

1. The model was chosen because Vestige looks good on it.
2. Only Vestige got a write / coordination tool.
3. The causal edge was planted, so the memory architecture did no work.
4. Failed competitor cells were dropped.
5. Some scores have no transcripts behind them.

## What the files show

**GPT-5.6 Sol is complete.** 5 trials × 3 arms × 3 agents = 45 transcripts,
all on disk. Five distinct live keys. Memory layer alive on every RAG and
Vestige cell. No memory 0/5 (all five fleets agreed on a wrong key). Dense
RAG 0/5 (four agreed wrong, one split). Vestige 5/5. The first
`vestige_backfill` contained the live key 15/15. The first `rag_search`
contained it 0/15. `agents_used_vestige_log` is 3 on every Vestige fleet and
0 on every no-memory and RAG fleet.

**Kimi K3 is complete except three missing transcript sets.** SuperMemory
5/5, transcripts present. Vestige is 4/4 if you only count cells whose
transcripts exist. The arm JSON for `runB-trial-1` lists Vestige 5/5, but
`transcript-sync-a{0,1,2}.json` are not in that folder. The same hole exists
for that trial’s no-memory and RAG arms. Hindsight on Kimi is 0/3, all
splits. No dead memory layers in the cells that remain.

**Excluded cells are published, not deleted.**
`results/WITHHELD-contaminated/README.md` documents three harness failures:
Zep runB-2 served the previous trial’s live key because FalkorDB was not
flushed; mem0 runB-1 leaked through `~/.mem0`; Hindsight runB-3 returned
0/9 retrievals (timeout). Those measurements are of the harness. They are
not in either model table. The arms were fixed and re-run; the clean cells
are in `results/`.

**The production oracle is not in the agent repo.** Any key in the keyring
turns the local suite green. That is true for every arm. It is why a
no-memory fleet can look like a win until replay.

## What is still unequal

**Tool sets.** `harness/agent/fleet_runner.py` gives Vestige
`vestige_backfill` and `vestige_log`. RAG, mem0, SuperMemory, Hindsight, and
Zep get a search tool only. No-memory gets neither. Giving RAG the same
write bus (`rag-bus` in [`ABLATION.md`](ABLATION.md)) scored 0/5, and the
Vestige transcripts show the live key in the first backfill payload before
any log. The bus is not a sufficient explanation. The tools are still not
the same. This benchmark does not isolate “memory backend” as the only
changed variable.

**The causal edge is authored by the harness.** `prepare_trial.py` tags the
cause memory and the failure with a shared `active_key` entity. The
measurement is traversal of that edge, not discovery of one. Removing it
(`sync-noedge`) scored 0/5 with `causes: []`. See [`ABLATION.md`](ABLATION.md).

**The models are not interchangeable.** Kimi dense RAG is 3/4 correct.
GPT-5.6 Sol dense RAG is 0/5. A pooled “RAG 4/23” number mixes those.

**SuperMemory passed 5/5 on Kimi.** Vestige is not the only successful
memory arm on that model. What still separates Vestige there is the first
call: 12/12 vs SuperMemory 1/15.

**Hindsight is n=3 and Zep is n=2.** Those rows are real and small.

**The original multi-model sweep was not pre-registered.** The ablation in
[`PREREGISTRATION.md`](PREREGISTRATION.md) was. The commit dates are the
audit trail.

## What is the same across arms

- The failing test, the 50-key keyring, the blank configs, and the
  production replay.
- The live key is in no file the agents can read.
- Competitor fleets did not receive `vestige_log` (field
  `agents_used_vestige_log` in each arm JSON).
- Contaminated and dead-backend cells were withheld with a README, then
  re-run.

Event-anchoring, the causal edge, and the coordination bus together are
load-bearing on GLM-5.2 (three ablation arms, 0/5 each, against a 5/5 full
Vestige run on the same keys). That write-up is [`ABLATION.md`](ABLATION.md).
