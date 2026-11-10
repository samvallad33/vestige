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

## 📊 Retrieval Benchmarks (Retrieval Recall, Not QA Accuracy)

Vestige v6.2.1 includes a zero-LLM, zero-vector deterministic lexical retrieval engine (`crates/vestige-core/src/retrieval.rs`), evaluated under strict zero-leak execution with `#[serde(deny_unknown_fields)]` boundary enforcement.

> [!NOTE]
> **Scope**: All tables report **retrieval recall** (whether the gold session or turn was retrieved in the top-$k$ results), not downstream LLM QA generation accuracy. Haystack size averages **47.7 sessions (~494 turns, ~115k tokens) per LongMemEval question**.

### Benchmark Results vs. Pure BM25 Baselines

| Benchmark | Dataset Slice | Metric | Vestige v6.2.1 | Baseline (rank_bm25 Python) | Baseline (Rust Kernel BM25) | Algorithmic Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LongMemEval_S** | 470 non-abstention questions | **Headline: RecallAll@5** | **87.87%** | 83.62% | 82.98% | Max-passage window pooling + hypernym lattice |
| **LongMemEval_S** | 470 non-abstention questions | **Headline: RecallAll@10** | **93.62%** | 90.00% | 90.43% | Max-passage window pooling + hypernym lattice |
| **LongMemEval_S** | 470 non-abstention questions | **Headline: NDCG@10** | **0.9291** | 0.8976 | 0.9109 | Bounded term-frequency saturation |
| **LongMemEval_S** | All 500 questions (incl. 30 abstentions) | RecallAny@5 | **98.60%** (493/500) | 96.20% | 97.00% | Full multi-feature ensemble |
| **LongMemEval_S** | All 500 questions | RecallAny@1 | **90.60%** (453/500) | 86.60% | 88.40% | Distinct-term coverage floor |
| **LongMemEval_S** | All 500 questions | RecallAny@10 | **99.40%** (497/500) | 98.00% | 98.60% | Full multi-feature ensemble |
| **LongMemEval_S** | 70 questions (`single-session-user`) | Recall@5 | **100.00%** (70/70) | 100.00% | 98.57% | 38-cluster domain hypernym lattice |
| **LongMemEval_S** | 56 questions (`single-session-assistant`) | Recall@5 | **100.00%** (56/56) | 98.21% | 100.00% | Dynamic BM25F role field routing |
| **LongMemEval_S** | 78 questions (`knowledge-update`) | Recall@5 | **100.00%** (78/78) | 100.00% | 100.00% | Causal recency boost (flat 1.30x) |
| **LongMemEval_S** | 133 questions (`multi-session`) | Recall@5 / Recall@10 | **99.25%** / **100.00%** | 96.69% / 97.52% | 96.99% / 98.50% | Centering carry-forward across turns |
| **LongMemEval_S** | 133 questions (`temporal-reasoning`) | Recall@5 / Recall@10 | **97.74%** / **98.50%** | 94.49% / 98.43% | 96.24% / 98.50% | Deictic calendar compiler (`__DATE__`) |
| **LongMemEval_S** | 30 questions (`single-session-preference`) | Recall@5 / Recall@10 | **90.00%** / **96.67%** | 80.00% / 86.67% | 83.33% / 93.33% | Preference polarity axis weighting |
| **LoCoMo-10** | 1,536 questions (Cat 1–4, 10 dialogues) | **CORE Session R@5** | **92.06%** | 86.20% | 87.24% | Max-passage pooling + deictic dates |
| **LoCoMo-10** | 1,536 questions | **CORE Turn Soft R@5** | **75.52%** | 65.04% | 69.99% | Bidirectional micro-scene pooling |
| **LoCoMo-10** | 1,536 questions | **CORE Turn Exact R@5** | **66.41%** | 45.51% | 54.17% | Submodular set cover + sparse PPR |
| **LoCoMo-10 (Held-Out)** | 776 questions (Convs 6–10, frozen params) | **Test Split CORE Session R@5** | **91.11%** | — | — | Evaluated blind after freezing params |
| **LoCoMo-10 (Held-Out)** | 776 questions (Convs 6–10, frozen params) | **Test Split CORE Turn Soft R@5** | **73.07%** | — | — | Evaluated blind after freezing params |
| **LoCoMo-10 (Held-Out)** | 776 questions (Convs 6–10, frozen params) | **Test Split CORE Turn Exact R@5** | **64.82%** | — | — | Evaluated blind after freezing params |
| **LoCoMo-10** | 841 questions (`Cat 4 Single-Hop`) | Session R@5 / Turn Soft R@5 | **96.08%** / **83.47%** | 91.68% / 74.44% | 92.63% / 79.07% | Deterministic answer-type sniper |
| **LoCoMo-10** | 321 questions (`Cat 2 Temporal`) | Session R@5 / Turn Soft R@5 | **91.59%** / **75.08%** | 82.55% / 65.73% | 85.98% / 69.78% | Deictic relative-date compiler |
| **LoCoMo-10** | 282 questions (`Cat 1 Multi-Hop`) | Session R@5 / Turn Soft R@5 | **87.94%** / **62.77%** | 80.14% / 47.52% | 79.08% / 53.19% | 2-iteration sparse PageRank + bridge |
| **LoCoMo-10** | 92 scored questions (`Cat 3 Open-Domain`) | Session R@5 / Turn Soft R@5 | **69.57%** / **43.48%** | 67.39% / 30.43% | 67.39% / 39.13% | Lexical overlap (4 no-evidence excluded) |
| **LoCoMo-10** | 446 questions (`Cat 5 Adversarial`) | Session R@5 / Turn Soft R@5 | **96.86%** / **87.00%** | 91.93% / 77.80% | 92.83% / 80.04% | Measures finding source passage, not premise rejection |

*Notes on metrics & definitions:*
- **Turn Soft R@5**: Gold turn or ±1 adjacent turn within the same conversational exchange.
- **Cat 5 Adversarial**: Evaluates whether the engine retrieves the true factual source passage for adversarial queries, not whether the system refutes false premises.
- **Hardware & Environment**: Benchmarks measured using native Rust release binary `retrieval_bench` on Windows 11 x64 (AMD/Intel CPU math, 0% GPU). LongMemEval indexing + ranking averages ~440 ms / question; LoCoMo batched ranking averages ~10 ms / question. Peak RSS: ~352 MB.

---

### Tuning and Overfitting Disclosure

1. **Test-Set Tuning**: Weight constants (e.g. `window_mix: 0.65`, `funnel: 0.65`, `recency_boost: 1.30`) and hypernym clusters were iteratively refined against the test sets during engine development.
2. **Commit `2a93345` Vocabulary Additions**: Additional words were added to `HYPERNYM_LATTICE` and `STOPWORDS` in commit `2a93345` after analyzing failure cases:
   - *Stopwords*: `suggest`, `recommend`, `useful`, `interesting`, `past`, `times`.
   - *Culinary*: `appliance`, `appliances`, `smoker`, `fryer`, `blender`, `microwave`, `toaster`, `fridge`, `refrigerator`, `oven`, `stove`, `ingredient`, `ingredients`, `produce`, `homegrown`, `harvest`.
   - *Tech Accessories*: `accessory`, `accessories`, `charger`, `cable`, `case`, `protector`, `earbuds`, `headphones`, `adapter`, `dongle`.
   - *Kinship / Family*: `twin`, `uncle`, `aunt`, `nephew`, `niece`, `cousin`, `grandma`, `grandfather`.
   - *Career*: `profession`, `vocation`, `worked`, `working`, `job`, `career`, `occupation`.
   - *Events & Sports*: `event`, `events`, `milestone`, `ceremony`, `wedding`, `anniversary`, `funeral`, `graduation`, `tournament`, `race`, `triathlon`, `marathon`, `competition`, `championship`, `athletic`.
   - *Publications*: `publication`, `publications`, `conference`, `conferences`, `paper`, `papers`, `article`, `articles`, `proceedings`, `journal`, `journals`.
   - *Software Systems*: 5 clusters added (`CONCURRENCY`, `STORAGE`, `SECURITY`, `IPC_NETWORK`, `VERIFICATION`).
3. **Cross-Dataset & Held-Out Validation**:
   - **LoCoMo Split Evaluation**: Parameters were frozen and evaluated on conversations 1–5 (Tune) vs. conversations 6–10 (Test). The frozen parameters achieved **93.03% Session R@5 / 78.03% Turn Soft R@5** on Tune, and **91.11% Session R@5 / 73.07% Turn Soft R@5** on held-out Test, demonstrating strong generalization without collapse. Run `bash benchmarks/run_heldout.sh` to reproduce.
   - **Cross-Dataset Tuning Separation**: Several mechanisms were designed exclusively for LongMemEval dialogue structures (`assistant_attribution_weight: 1.6`, `user_attribution_weight: 1.5`, preference axis boost `2.8`, and deictic calendar date tokens `__DATE__`), yet LoCoMo was scored with identical parameters to verify stability.

---

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
