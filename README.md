<div align="center">

# VESTIGE

**/ˈvɛs.tɪdʒ/ · VESS-tij**

*From the Latin **vestīgium**, a physical footprint pressed into the earth, and the direct root of **investigate** (**in-vestīgāre**: to follow the track backward).*

### Deterministic, local-first memory and safety kernel for AI agents

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://github.com/samvallad33/vestige)
[![License: AGPL-3.0](https://img.shields.io/github/license/samvallad33/vestige)](LICENSE)
[![Release](https://img.shields.io/github/v/release/samvallad33/vestige)](https://github.com/samvallad33/vestige/releases)

No embeddings, no LLMs, no GPU, and no API keys. It runs locally on CPU.

</div>

---

## Why Vestige?

Roman trackers read *vestigia* in soil to reconstruct what passed through, how heavy it was, which way it moved, and how recently. Vestige does the same for agent memory. It records immutable footprints in an append-only log and walks the trail, instead of guessing meaning from a vector space.

Human memory systems solved truth, order, and provenance under hard limits for 5,000 years: Pāṇini's *anuvṛtti* carry-forward registers, *isnād* witness chains, Andean *khipu* top-cord rollups, Torrens title curtains, and the archaeological Harris matrix. Vestige takes those structural ideas onto a local Rust kernel.

---

## Architecture

- **Strata.** An append-only, hash-chained, signed log. Frames carry BLAKE3 payload digests and bi-temporal validity bounds (`valid_from_ms`, `valid_until_ms`). Admitted writes return receipts. Later claims supersede earlier ones; old facts stay on the log.
- **FSRS-6.** Spaced-repetition scheduling that lets unused noise fade.
- **RSB (Retroactive Salience Backfill).** The moat. *Cause never looks like the symptom.* A later turn can raise an earlier memory even when they share no words. Salience is set at write time, and new learning can raise or lower it later, a pattern described in Sheena Josselyn's engram research. In the birthday-dinner demo, a buried "felt off after Bangkok Garden" memory is raised once Maya's peanut allergy is recorded, so a later question about her birthday dinner can retrieve it.
- **Backfill.** Upstream causal walks from a failure or a later fact back to the earlier turn that caused it.
- **Rank-1 seating.** Deterministic slot-1 rules on the retrieved set, such as yielding an assistant reply to a covering user turn, with no vector index.
- **GhostLink.** Pairs recorded memories, each with a proof.
- **Operator hooks, one-use permits, and receipts.** `npx @vestige/init --guard` installs the Operator gate. A one-time owner permit lasts 15 minutes, is consumed once, and stops or allows write receipts.

---

## Benchmarks

These numbers measure retrieval: whether the right memory landed in the top k. They do not measure answer accuracy. They are the latest scores merged to `main` from [PR 758](https://github.com/samvallad33/vestige/pull/758) and [PR 759](https://github.com/samvallad33/vestige/pull/759) (`4de0f6ff`, `retrieval_bench` `sha256=24c3165e0c374c071c02ed5342fd1c7d68e14e108bfff0992862911e716d33ea`), verified with `--hash-check` (`0 / 500` differing questions across hash seeds), CPU only, with no embeddings, no LLM, no GPU, and no API keys.

| Benchmark | Level | Metric | Score |
| :--- | :--- | :--- | :--- |
| **LoCoMo-10** CORE (1,536 questions) | Session | R@1 / R@5 | **1133/1536 (73.76%)** / **1416/1536 (92.19%)** |
| **LoCoMo-10** CORE (1,536 questions) | Turn | Exact R@1 / R@5 / R@10 | **817/1536 (53.19%)** / **1129/1536 (73.50%)** / **1232/1536 (80.21%)** |
| **LongMemEval-S** official (470 non-abstention) | Session | Any@1 / Any@5 / Any@10 | **430/470 (91.49%)** / **463/470 (98.51%)** / **469/470 (99.79%)** |
| **LongMemEval-S** official (470 non-abstention) | Session | All@5 / All@10 | **422/470 (89.79%)** / **446/470 (94.89%)** |
| **LongMemEval-S** official (470 non-abstention) | Turn | Any@1 / Any@5 / Any@10 / All@5 | **360/470 (76.60%)** / **428/470 (91.06%)** / **441/470 (93.83%)** / **323/470 (68.72%)** |
| **BEAM 100K** (355 scored questions) | Session | Any@1 / Any@5 / Any@10 / All@5 | **133/355 (37.46%)** / **249/355 (70.14%)** / **275/355 (77.46%)** / **137/355 (38.59%)** |
| **BEAM 100K** (355 scored questions) | Turn | Any@1 / Any@5 / Any@10 / All@5 | **164/355 (46.20%)** / **241/355 (67.89%)** / **272/355 (76.62%)** / **125/355 (35.21%)** |
| **MemoryAgentBench** QA2 421K (100 questions) | Document | All@5 *(PR #695)* | **85/100 (85.00%)** |

Reproduce:

```bash
bash benchmarks/run_all.sh
```

Full tables and method notes are in [benchmarks/README.md](benchmarks/README.md).

---

## Quick start

```bash
npx @vestige/init --guard
```

That sets up local binaries and an MCP connection for any AI agent. `--guard` adds the Operator gate. The MCP command is `vestige-mcp`:

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

From source:

```bash
git clone https://github.com/samvallad33/vestige.git
cd vestige
cargo build --release --package vestige-mcp
```

More in [Getting started](docs/GETTING-STARTED.md).

---

## License

Built by [Sam Valladares](https://github.com/samvallad33) ([@samvallad33](https://github.com/samvallad33)). Licensed under [AGPL-3.0](LICENSE).

<div align="center">

*Follow the footprints.*

</div>
