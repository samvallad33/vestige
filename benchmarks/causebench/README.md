# CauseBench

CauseBench is a one-command, deterministic, offline benchmark for the causal-gap retrieval task.

It asks a retriever to find the older memory that caused a later failure when the cause and symptom do not look alike. Text resemblance baselines are adversarially given a lookalike memory, so they score 0% recall@1.

Run from the repository root:

```bash
bash benchmarks/causebench/run.sh
```

Contract:

- no API keys
- no network
- fixed seed, 424242
- Python standard library only
- expected numbers are checked by the harness before exit 0

Verified target numbers:

- Vestige causal bridge recall@1, synthetic: 60%
- Vestige causal bridge recall@1, real: 50%
- baselines recall@1, synthetic and real: 0%
