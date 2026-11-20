# Vestige — Deterministic Operating System & Safety Kernel for AI Agents

**Vestige is a local-first deterministic operating system, execution safety kernel, and zero-vector causal memory engine for autonomous AI agents built in Rust.**

Unlike probabilistic LLMs and vector RAG databases that rely on floating-point similarity without tracking causality, Vestige wraps AI agents in a local Rust binary. It enforces cryptographic execution gates, bi-temporal state tracking, and zero-vector causal retrieval with **$0 API cost** and **no GPU overhead**.

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://github.com/samvallad33/vestige)
[![License: AGPL-3.0](https://img.shields.io/github/license/samvallad33/vestige)](LICENSE)
[![Release](https://img.shields.io/github/v/release/samvallad33/vestige)](https://github.com/samvallad33/vestige/releases/tag/v6.2.3)

> 🚀 **Live WebGPU Companion (Zero Install):** [https://stara-companion.pages.dev/?v=pro-reset](https://stara-companion.pages.dev/?v=pro-reset)

---

## ⚡ Quick Start (`v6.2.3`)

### 1. Auto-configure for Claude Code, Cursor & Windsurf (npm):
```bash
npx @vestige/init --guard
```
*Detects installed AI agents, configures the MCP server, and installs the Operator safety guard.*

### 2. Standalone MCP Server:
```json
{
  "mcpServers": {
    "vestige": { "command": "npx", "args": ["-y", "vestige-mcp-server"] }
  }
}
```
*(Or in Antigravity: `agy plugin add vestige`)*

### 3. Pre-built Native Binaries (macOS ARM/x86, Linux, Windows):
Download pre-built standalone binaries directly from [GitHub Releases](https://github.com/samvallad33/vestige/releases/tag/v6.2.3), or install via shell:
```bash
curl -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/scripts/install.sh | sh
```

---

## 📊 How Vestige Compares

| Capability | **Vestige (`samvallad33/vestige`)** | Vector RAG Memory *(Mem0, Zep, Letta)* | Policy Wrappers *(Microsoft Agent OS, Kern)* |
| :--- | :--- | :--- | :--- |
| **System Category** | **Deterministic Agent OS & Causal Kernel** | Probabilistic Vector Memory Store | Stateless Middleware / Policy Firewall |
| **Retrieval Engine** | **Zero-Vector Causal Math** (RSB v4, Rank-1 seating, FSRS-6) | Dense Embeddings + ANN Cosine Similarity | None / External RAG Dependency |
| **Compute & API Cost** | **$0 / CPU-Only** (verified on 2011 Core i5-2520M) | GPU / Cloud Embedding & Reranker APIs | Low CPU (Policy Rules Only) |
| **State & Audit Ledger** | **Strata**: Append-only, signed, BLAKE3 hash-chained, bi-temporal | Mutable DB rows / overwrites | Append-only audit logs (no causal memory graph) |
| **Execution Governance** | **Out-of-Band Operator Gates (`--guard`)** + 15-min owner permits | None (relies on agent self-restraint) | Pre-execution policy & sandbox checks |
| **Root-Cause Tracing** | **`causal_walk` & `vestige prove`** (automated git bisect) | Not supported | Log inspection only |
| **Multi-Agent Sync** | **Reflex Context Manifest + Stale-Belief Push** | Shared vector namespace (stale drift risk) | Per-process isolation |

---

## 🛠️ Core Kernel Subsystems

* **Strata (Cryptographic Causal Ledger):** Append-only, signed, BLAKE3 hash-chained log providing bi-temporal validity bounds (`valid_from_ms`, `valid_until_ms`). Writes return receipts; new claims supersede older ones while historical facts remain immutable.
* **Operator Gates & Reflex Hooks (`--guard`):** Out-of-band execution governance requiring 15-minute single-use owner permits for high-risk actions, logging every block and allowance to a dedicated ledger.
* **Zero-Vector Causal Retrieval (RSB v4 + Rank-1 Seating + FSRS-6):** Structural provenance engine using Retroactive Salience Backfill, deterministic slot-1 seating rules, and FSRS-6 memory decay instead of embeddings. *Cause never looks like the symptom* — a later turn raises earlier relevant context even with zero shared words.
* **Causal Root-Cause Proofs (`causal_walk` & `vestige prove`):** Traces upstream failure and commit edges, running deterministic git-bisect verification against your own tests with receipted runs.
* **Multi-Agent Context Manifest & Compaction Survival:** Manages context token budgets, pushes live stale-belief revocations across subagents, and restores decisions across context compactions.

---

## 📈 Benchmarks (`v6.2.3`)

Benchmarked using the local `retrieval_bench` binary on an Intel Core i5-2520M with 8 GB RAM, CPU only (**no embeddings, no LLM, no GPU, $0 API costs**):

| Benchmark | Split / Slice | Level | Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **LoCoMo-10** | Blind Convs 6–10 (`n = 985`, out-of-sample) | Session / Turn | Sess R@5 / Turn Exact R@5 / Turn R@10 | **93.71%** / **80.41%** / **88.02%** (`~59ms/q`) |
| **LoCoMo-10** | All 10 Convs (`n = 1,982`) | Session / Turn | Sess R@5 / Turn Exact R@5 / Turn Soft R@5 | **94.25%** / **81.84%** / **84.71%** |
| **LongMemEval-S** | Official (`470` non-abstention) | Session | Any@1 / Any@5 / Any@10 / CappedAll@5 | **85.74%** / **96.17%** / **99.57%** / **92.08%** |
| **BEAM 100K** | `355` scored questions (`2,127` sub-sessions) | Session / Turn | Pkt Sess Any@5 / Pkt Turn Any@5 / Temporal Any@5 | **78.59%** / **70.42%** / **100.0%** |
| **MemoryAgentBench** | QA2 421K (`100` questions) | Document | All@5 | **85/100 (85.00%)** |

Reproduce locally:
```bash
bash benchmarks/run_all.sh
```

Full tables and methodology are in [benchmarks/README.md](benchmarks/README.md).

---

## License

Built by [Sam Valladares](https://github.com/samvallad33) ([@samvallad33](https://github.com/samvallad33)). Licensed under [AGPL-3.0](LICENSE).
