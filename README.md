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

These measure retrieval, meaning whether the right memory came back in the top 5. They do not measure answer accuracy.

| Benchmark | Vestige | Plain BM25 |
| :--- | :--- | :--- |
| LongMemEval_S, any gold session in top 5, all 500 questions | 98.60% | 97.00% |
| LongMemEval_S, official all gold sessions in top 5, 470 questions | 87.87% | 82.98% |
| LoCoMo, right session in top 5, 1,536 questions | 92.06% | 87.24% |
| LoCoMo, exact turn in top 5 | 66.41% | 54.17% |

This is the top retrieval result without embeddings we know of on LongMemEval_S. Settings were tuned on these datasets. As a split check, LoCoMo conversations 6 to 10 still score 91.11% session recall at 5.

On a 2011 dual core i5 laptop with 8 GB of RAM, a LongMemEval question takes about 390 ms and a LoCoMo question about 9 ms, with peak memory around 335 MB.

Rerun everything with one command. Full tables and disclosures are in [benchmarks/README.md](benchmarks/README.md).

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
