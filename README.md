<div align="center">

# VESTIGE v6.2.1
### The Ring-0 Autonomic Execution Kernel & Causal Proof Engine for AI Agents

**Think before doing. Prove before saving. Memory is root cause, not search.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Rust: 2024](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://www.rust-lang.org)
[![Embeddings: 0 Vectors](https://img.shields.io/badge/Embeddings-0_Vectors-brightgreen.svg)](#-zero-vector-architecture)
[![IPC Latency: <0.2ms](https://img.shields.io/badge/IPC_Latency-<0.2ms-blueviolet.svg)](#-autonomic-reflex-hooks)
[![Operator: 102 Rules](https://img.shields.io/badge/Gate_Rules-102_Deterministic-red.svg)](operator/README.md)
[![Benchmarks: 98.60% R@5](https://img.shields.io/badge/LongMemEval_S-98.60%25_R@5-success.svg)](benchmarks/README.md)

[**Quick Start**](#-quick-start) · [**Benchmarks**](#-audited-retrieval-benchmarks) · [**Architecture**](#-architecture-the-closed-loop-kernel) · [**MCP Tools**](#-the-6-mcp-verbs) · [**Reflex Hooks**](#-autonomic-reflex-hooks) · [**CLI**](#-host-cli)

</div>

---

## 🏆 Audited Retrieval Benchmarks

Vestige v6.2.1 integrates the audited deterministic CPU retrieval engine (`crates/vestige-core/src/retrieval.rs`), evaluated under blind zero-leak execution with `#[serde(deny_unknown_fields)]` boundary enforcement:

| Benchmark | Dataset / Slice | Metric | Vestige v6.2.1 | Baseline (BM25) | Execution Guarantee |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LongMemEval_S** | 500 probes (~115K tok/probe) | **Overall Recall@5** | **98.60%** (493/500) | 68.40% | 8.21 ms / probe |
| **LongMemEval_S** | 500 probes | **Overall Recall@1** | **90.60%** (453/500) | 48.20% | 78 MB peak RAM |
| **LongMemEval_S** | 500 probes | **Overall Recall@10** | **99.40%** (497/500) | 76.80% | Pure CPU Math |
| **LongMemEval_S** | 500 probes | **NDCG@10** | **0.9259** | 0.5840 | 0% GPU / $0.00 cost |
| **LongMemEval_S** | 70 questions (`single-session-user`) | Recall@5 | **100.00%** (70/70) | 64.28% | 37-cluster hypernym lattice |
| **LongMemEval_S** | 56 questions (`single-session-assistant`) | Recall@5 | **100.00%** (56/56) | 69.64% | Dynamic BM25F field routing |
| **LongMemEval_S** | 78 questions (`knowledge-update`) | Recall@5 | **100.00%** (78/78) | 72.10% | Causal supersession priority |
| **LongMemEval_S** | 133 questions (`multi-session`) | Recall@10 / R@5 | **100.00%** / **99.25%** | 74.40% | Micro-scene carry-forward |
| **LongMemEval_S** | 133 questions (`temporal-reasoning`) | Recall@5 | **97.74%** (130/133) | 60.15% | Deictic calendar compiler |
| **LoCoMo-10** | 1,986 dialogues (35+ sessions/convo) | **Core Session R@5** | **91.93%** | 64.20% | 2-iteration sparse PPR |
| **LoCoMo-10** | 1,986 dialogues | **Core Turn Soft R@5** | **75.39%** | 42.10% | Submodular set cover rerank |
| **LoCoMo-10** | 841 dialogues (`Cat 4 Single-Hop`) | Session R@5 | **95.84%** | 71.30% | Answer-type morphology sniper |
| **LoCoMo-10** | 446 dialogues (`Cat 5 Adversarial`) | Session R@5 | **96.86%** | 72.80% | Exact speaker attribution |
| **LoCoMo-10** | 321 dialogues (`Cat 2 Temporal`) | Session R@5 | **91.28%** | 62.40% | ISO calendar compilation |
| **LoCoMo-10** | 282 dialogues (`Cat 1 Multi-Hop`) | Session R@5 | **87.94%** | 58.10% | Rare bridge propagation |

*Reproduce locally in one command: `node benchmarks/run_stage1_retrieval.mjs`. See [benchmarks/README.md](benchmarks/README.md).*

---

## ⚡ Quick Start

```bash
# 1. Install daemon, wire reflex hooks into your agent harness, and enable the pre-tool gate
npx @vestige/init --guard

# 2. Start the MCP server + Unix Domain Socket IPC daemon (~/.vestige/reflex.sock)
vestige serve

# 3. Verify cryptographic integrity of the append-only Strata log
vestige strata-verify ~/.vestige/strata
```

---

## 🧠 Architecture: The Closed-Loop Kernel

Vestige decouples deliberative agent reasoning from deterministic execution safety. The 6 MCP tools form the high-level cognitive interface, while Reflex hooks run autonomically on the host in `<0.2ms`:

```text
 ┌────────────────────────────────────────────────────────┐
 │            Deliberative Agent Prefrontal Cortex        │
 │    session · remember · recall · causal_walk · curate  │
 └───────────────────────────┬────────────────────────────┘
                             │ stdio / JSON-RPC (MCP)
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           vestige-mcp Kernel Daemon (Port 3928)        │
 │  ┌──────────────────────────────────────────────────┐  │
 │  │ Hot In-Memory Strata Graph                       │  │
 │  │ • Exact Handle & Symbol Index (BM25F + PPR)      │  │
 │  │ • FSRS-6 Stability Decay & Precision Forgetting  │  │
 │  │ • Retroactive Salience Backfill (RSB DAG Engine) │  │
 │  └──────────────────────────┬───────────────────────┘  │
 └─────────────────────────────┼──────────────────────────┘
           Unix Domain Socket  │  Exclusive Append Lock
       (~/.vestige/reflex.sock)│
                               ▼
 ┌───────────────────────────┐ ┌──────────────────────────┐
 │ Autonomic Reflex Hooks    │ │ Append-Only Strata Log   │
 │ • SessionStart (Context)  │ │ • BLAKE3 Hash Chaining   │
 │ • PreToolUse (102 Rules)  │ │ • Ed25519 Segment Seals  │
 │ • PostToolUse (Auto-RSB)  │ │ • Non-Erasable Receipts  │
 │ • PreCommit (Proof Gate)  │ └──────────────────────────┘
 └───────────────────────────┘
```

---

## 🛠️ The 6 MCP Verbs

Vestige condenses memory and execution management into 6 flat-JSON cognitive tools:

| Tool | Action Discriminator | Purpose & Codebase Anchors |
| :--- | :--- | :--- |
| **`session`** | `start`, `status`, `set_intention`, `check_intentions`, `complete_intention`, `project` | Boot turn context, manage prospective intentions, check runtime health, and project fenced rules into `CLAUDE.md` / `AGENTS.md`. |
| **`remember`** | `save`, `backfill` | Single gated write. Anchors records to `repo_path` and `files`, triggers inline Retroactive Salience Backfill (RSB), and recompiles rule fences when tagged `"pinned"`. |
| **`recall`** | `lookup`, `exact` | Exact handle resolution (`id`, `8+ char prefix`, `tag`) or token-exact BM25. Refusal returns candidate `discovered_handles` to prevent hallucinated lookups. |
| **`causal_walk`**| `walk`, `prove` | Traverses explicit graph edges (`touched`, `closed_by`, `corrects`, `derived_from`, `evidence_of`) back to root causes; runs deterministic git-bisect witnesses via `prove`. |
| **`curate`** | `promote`, `demote`, `edit`, `link`, `bridge`, `connect_identities`, `suppress` | Modifies graph topology and FSRS-6 weights. Enforces causal supersession, bridges multi-hop entities, and suppresses invalid records without erasing log bytes. |
| **`precedent`** | `stops`, `receipt`, `replay`, `verify_log`, `scan_secrets`, `selftest` | Audits Operator guard stops, inspects BLAKE3/Ed25519 write receipts, replays execution traces, scans secrets, and verifies cryptographic log continuity. |

---

## ⚡ Autonomic Reflex Hooks

Reflex runs on the host via `~/.vestige/reflex.sock` in **0 LLM turns** and **<0.2ms**:

1. **`SessionStart`**: Injects hot entity handles and pinned rules into context at Turn 1 and after context compaction.
2. **`PreToolUse`**: Gated by **102 deterministic Operator rules** (OP-000 through OP-114, reserving OP-086..097, OP-100). Intercepts obfuscated shell attacks, recursive deletions, credential dumps, and warns if a touched file has an unresolved `failure` node in Strata.
3. **`PostToolUse`**: On command failure (`exit_code != 0`), automatically writes a signed failure frame, triggers RSB credit assignment across upstream touched files, and surfaces decayed lessons (`retrievability < 0.5`).
4. **`PreCommit`**: Enforces proof-carrying commits by verifying that a deterministic bisect test witness exists for recent failure fixes.

---

## 🖥️ Host CLI

The `vestige` host binary handles boundary lifecycle tasks:

```bash
vestige serve                         # Run MCP server + reflex.sock IPC daemon
vestige dashboard [--port 3927]       # Open local loopback UI (http://127.0.0.1:3927)
vestige backup <output-dir>           # Seal active Strata log and copy to backup directory
vestige strata-verify <dir>           # Cryptographically verify BLAKE3 frames and Ed25519 keys
vestige health                        # Run storage diagnostics and retention health checks
vestige stats [--tagging] [--states]  # Display memory state distributions and FSRS-6 stats
vestige prove                         # Run frozen-protocol git bisect verification
```

---

## 🔒 Zero Vector Architecture

- **Zero Vectors:** Exact handle indexing, BM25F field scoring, 37-cluster hypernym expansion, and 2-iteration sparse PageRank replace high-overhead vector databases.
- **Zero Cloud Leak:** Release builds exclude `reqwest` and cloud SDKs. All graph operations, safety classifications, and benchmark evaluations run 100% locally on CPU.
- **Append-Only Integrity:** Deletions return `unavailable_in_4_0`; records are demoted or suppressed with audit receipts, preserving immutable audit trails.

---

## 📚 Reference & Documentation

- [Architecture & Storage Log (Strata)](docs/STORAGE.md)
- [Operator Pre-Tool Gate (102 Rules)](operator/README.md)
- [Cognitive Sandwich Protocol](docs/COGNITIVE_SANDWICH.md)
- [Tool Contracts & MCP Schemas](docs/TOOL-CONTRACTS.md)
- [Getting Started Guide](docs/GETTING-STARTED.md) · [FAQ](docs/FAQ.md) · [Upgrading](docs/UPGRADING.md)

Built from first principles in Rust by Sam Valladares. Licensed under AGPL-3.0.
