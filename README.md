<div align="center">

# Vestige
### Memory and a safety gate for AI coding agents.

</div>

---

## What it does

- **Remembers.** Agents keep what happened in earlier sessions and can trace a failure back to the change that caused it.
- **Stops bad tool calls.** 102 fixed rules check every tool call before it runs and block things like recursive deletes and credential dumps.
- **Asks again.** An earlier yes is not a yes now. A one-time owner permit lasts 15 minutes and then the agent has to ask again.
- **Proves it.** Every write is signed and hash chained, so anyone can check the log offline.

It runs on your machine on a plain CPU. No LLM, no GPU, no cloud calls.

---

## Quick start

```bash
npx @vestige/init --guard
vestige serve
```

---

## Benchmarks

These measure retrieval, meaning whether the right memory came back in the top 5. They do not measure answer accuracy. Every number is evaluated on 100% of the benchmark questions with zero embeddings, zero vectors, and zero cloud LLM calls ($0.00 spend).

| Benchmark | Scale & Setting | Vestige v6.2.1 | Plain BM25 |
| :--- | :--- | :--- | :--- |
| **LongMemEval_S** | Official all gold sessions in top 5 (470 non-abs) | **91.28%** | 82.98% |
| **LongMemEval_S** | Any gold session in top 5 (all 500 questions) | **98.60%** | 97.00% |
| **LongMemEval_M** | Scaled haystack (~476 sessions, ~4,894 turns/Q; 470 non-abs) | **91.70%** Any@5 / **76.38%** All@5 | — |
| **LoCoMo-10** | Full conversational dialogues, right session in top 5 (1,536 CORE Qs) | **92.25%** | 87.24% |
| **LoCoMo-10** | Conversational turn-level soft recall in top 5 (1,536 CORE Qs) | **78.13%** | 69.99% |
| **MemoryAgentBench** | Multi-hop reasoning, 421K distractor context (`ruler_qa2`, 100 Qs) | **98.00%** Any@5 / **93.00%** All@10 | — |
| **MemoryAgentBench** | Needle retrieval in 197K context (`ruler_qa1`, 1,204 docs, 100 Qs) | **95.00%** R@5 / **99.00%** R@10 | — |
| **MemoryAgentBench** | Merged 400K scenarios (`longmemeval_s*`, 5 sets, 287 non-abs) | **94.08%** Any@5 / **85.02%** All@5 | — |
| **BEAM 100K** | 20 conversations, 2,127 sub-sessions, right session in top 5 (355 Qs) | **66.20%** Any@5 / **74.93%** Any@10 | — |
| **BEAM 100K** | Turn exact ID recall in top 5 (355 scored questions) | **45.92%** Any@5 / **55.77%** Any@10 | — |

This is the top retrieval result without embeddings we know of on LongMemEval_S. Settings were tuned on these datasets. As a blind split check, LoCoMo held-out conversations 6 to 10 score **91.75%** session recall at 5 and **76.03%** turn soft recall at 5.

On a plain CPU with 8 GB of RAM, a LongMemEval question takes ~500 ms, a LoCoMo question ~7 ms, and a 5,000-turn LongMemEval_M haystack ~5 seconds, with peak RSS under 350 MB.

Full tables, category breakdowns, and zero-leak disclosures are in [benchmarks/README.md](benchmarks/README.md).

```bash
bash benchmarks/fetch_data.sh && bash benchmarks/run_all.sh
```

---

## Learn more

- [Getting started](docs/GETTING-STARTED.md)
- [Safety gate rules](operator/README.md)
- [Storage and signed log](docs/STORAGE.md)
- [Tool reference](docs/REFERENCE.md)
- [FAQ](docs/FAQ.md)

---

Built in Rust by Sam Valladares. Licensed under AGPL-3.0.
