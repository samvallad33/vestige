# Vestige

**Local-first memory for AI agents that finds the cause, not just the match.**

Vestige remembers your decisions, catches contradictions before they cost you, and traces a failure back to the older memory that actually caused it. One 25MB Rust binary over MCP. No cloud, no API keys, no telemetry. Your data never leaves your machine.

[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/badge/tests-1961_passing-22c55e)](https://github.com/samvallad33/vestige/actions)
[![Binary](https://img.shields.io/badge/binary-25MB_single_file-informational)](https://github.com/samvallad33/vestige/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6)](LICENSE)

[Consulting](#-consulting--core-infrastructure-advisory) · [Install](#install) · [Why not RAG](#why-not-just-rag) · [Benchmark](#the-receipts-silent-rotation) · [Science](#the-science) · [Tools](#the-14-tools) · [Dashboard](#the-dashboard) · [Pro](#vestige-pro) · [Docs](#go-deeper)

## 💼 Consulting & Core Infrastructure Advisory

Autonomous agents are currently bleeding enterprise budgets via prompt bloat and context window amnesia.

I take on a limited number of technical advisory retainers and consulting projects for AI developer tool startups, multi-agent frameworks, and enterprise engineering teams looking to optimize their context economics.

### Core Specializations:
* **Context Optimization & Filtering:** Implementing local Prediction Error Gating to strip out redundant tool runtime noise and drop token overhead by 40%–60%.
* **Causal Agent Memory Design:** Structuring local SQLite graph architectures using Retroactive Salience Backfilling to eliminate agent amnesia during heavy, multi-file code execution.
* **Air-Gapped AI Governance:** Designing zero-knowledge, high-performance Rust memory scaffolding that runs entirely on local metal to protect proprietary enterprise IP.

For architectural reviews, integration advisory, or founding infrastructure roles, reach out directly at: **sam@vestige.sh**

---

<p align="center">
  <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">
    <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/black-box-cause.gif" alt="Vestige Black Box: a SIGSEGV on startup traced back to a version pin set 23 days earlier, with the receipt" width="100%">
  </a>
</p>

<p align="center"><sub><b>A labeled fixture store, a real run.</b> The incident is fictional, seeded into a Vestige store with a seven month backdated timeline. The engine is not. A SIGSEGV on startup in an arm64 container, and the version pin set 23 days earlier that shares zero words with the failure. Similarity ranked the pin fourth. Backfill reached back, ranked it first, persisted the causal edge, and sealed the receipt. It names the suspects. It never calls the verdict. <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">Watch the full 58 second walk</a>, then run <code>vestige backfill --contrast</code> on your own store.</sub></p>


Agents re-learn the same lessons: they recommend a change you already tested and rejected, re-derive a fix that was already written down, and treat every session as if the last one never happened. Vestige is the memory layer that ends that. Any MCP-capable agent (Claude Code, Claude Desktop, Codex, Cursor, and others) writes memories as you work and retrieves them later, modeled on real cognitive science: redundant memories merge, contradicted ones are flagged, unused ones fade, and when a failure hits, Vestige reaches **backward** to the decision that set it up.

The cause never looks like the bug. That is the whole product.

## Install

You need Node.js. No Docker, no signup, no compile step (prebuilt for macOS ARM + Intel, Linux x86_64, Windows x86_64).

Android (Termux) builds from source today; see [docs/INSTALL-TERMUX.md](docs/INSTALL-TERMUX.md).

```bash
npm install -g vestige-mcp-server@latest
```

Connect it to your agent. Every MCP client understands this config:

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

| Client | Setup |
|---|---|
| Claude Code | `claude mcp add vestige vestige-mcp -s user` |
| Codex | `codex mcp add vestige -- vestige-mcp` |
| Cursor / VS Code / Windsurf | [docs/integrations/](docs/integrations/) |
| Claude Desktop | [docs/CONFIGURATION.md](docs/CONFIGURATION.md#claude-desktop-macos) |
| Cline / Continue / Zed / Goose | the JSON above, in that client's MCP settings |

Verify: `vestige dashboard`, then open **http://localhost:3927/dashboard**. First run downloads a 130MB embedding model and, in the background, a ~150MB reranker, once; after that Vestige is fully offline, forever. Full walkthrough: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

## Why not just RAG?

RAG retrieves text that resembles the query. That is the right tool when the answer looks like the question, and the wrong tool when the cause of a problem looks nothing like the symptom: a config choice from three weeks ago, a library pin, an assumption nobody flagged as risky.

| | Vector search | Vestige |
|---|---|---|
| Retrieval basis | Similarity to the query | Causal + temporal links, plus similarity |
| Root cause of a failure | Cannot; the cause does not resemble the bug | `vestige backfill --contrast` reaches backward to it |
| Contradictions | Both stored, both returned | Detected and flagged (`claim_contradicts_memory`) |
| Redundant writes | Accumulate | Merged on write (prediction-error gating) |
| Unused memories | Persist at full weight | Fade (FSRS-6 spaced repetition) |
| Your data | Usually a cloud service | Never leaves your machine |

The backward reach implements **Retroactive Salience Backfill** (Zaki, Cai et al., *Nature* 2024, 637:145-155, [DOI 10.1038/s41586-024-08168-4](https://doi.org/10.1038/s41586-024-08168-4)): when a memory turns out to matter, the salience of the earlier memories that led to it is raised, so the causal chain becomes retrievable even though the surface text never matched. Every backfill result ships with a receipt naming the exact evidence path; Vestige reports receipt-backed candidate causes, never an unverifiable verdict.

And the limitation on the left column is not marketing: DeepMind proved single-vector retrieval mathematically incapable of certain relevance patterns ([arXiv:2508.21038](https://arxiv.org/abs/2508.21038), ICLR 2026).

## The receipts: Silent Rotation

The claim is testable, and the test ships with all 246 agent transcripts it produced. Three coding agents fix one failing e2e test; the fix needs the currently live signing key id, randomized per trial from a 50-key keyring, present in no file the agents can read. It exists only in the memory layer. The dangerous outcome is converging on a planted decoy: tests pass, the merge is clean, production breaks.

| Arm (6 models, 25 trials) | Converged correct | Converged **wrong** | Split |
|---|---|---|---|
| No memory | 0/25 | **21/25** | 4/25 |
| Dense cosine RAG | 4/23 | **12/23** | 7/23 |
| Vestige | 20/23 | **0/23** | 3/23 |

On the verbatim queries the agents typed, the causal memory ranks 7th of 8 under both dense cosine and BM25 while the decoy ranks 1st. Reproduce the central measurement in two seconds, stdlib only:

```bash
git clone -b benchmark/silent-rotation --depth 1 https://github.com/samvallad33/vestige.git
cd vestige/benchmarks/silent-rotation
python3 tests/bm25_baseline.py results/runA-trial-1/corpus-export.json --no-dense
```

The caveats are published alongside the results, including the trials a plain cosine baseline ties and the trial Vestige loses.

## The science

Every mechanism is a cited result, implemented in Rust, running locally. Full write-up: [docs/SCIENCE.md](docs/SCIENCE.md).

| Mechanism | What it does | Source |
|---|---|---|
| Prediction-Error Gating | Stores only the novel; merges redundant, flags contradictory | Hippocampal novelty gating |
| FSRS-6 spaced repetition | Used memories persist, unused ones fade | Modern spaced-repetition research |
| Retroactive Salience Backfill | Reaches backward to a failure's root-cause memory | Zaki, Cai et al. 2024, *Nature* |
| Synaptic Tagging | Marks memories for later consolidation | Frey & Morris 1997 |
| Spreading Activation | One retrieval activates related memories through the graph | Collins & Loftus 1975 |
| Dual-Strength | Storage strength vs retrieval strength, tracked separately | Bjork & Bjork 1992 |
| Memory Dreaming | Sleep-like replay and synthesis | Sleep consolidation research |
| Active Forgetting | Reversible top-down suppression, cascading to neighbors | Anderson 2025, Davis 2020 |

## The tools

Your agent calls these; you rarely do.

| Tool | Purpose |
|---|---|
| `recall` | Retrieve memories relevant to the current context |
| `smart_ingest` | Store a fact, gated for novelty and contradiction |
| `backfill` | Reach backward from a failure to its candidate cause |
| `receipt` | Inspect retrieval receipts and evidence replay ([guide](docs/DECISION_RECEIPTS.md)) |
| `project` | Project durable decisions, patterns and rules into CLAUDE.md or MEMORY.md, one memory id per line ([guide](docs/PROJECTION.md)) |
| `memory` · `graph` · `intention` | Inspect, promote, explore, track goals |
| `maintain` · `dedup` · `suppress` | Consolidation, merge, reversible forgetting |
| `memory_status` · `codebase` · `source_sync` · `session_start` | Health, code index, connectors, session priming |

Project scoping, hygiene workflows, and making memory a standing habit for your agent: [docs/MEMORY_HYGIENE.md](docs/MEMORY_HYGIENE.md) · [docs/AGENT-MEMORY-PROTOCOL.md](docs/AGENT-MEMORY-PROTOCOL.md) · [docs/CLAUDE-SETUP.md](docs/CLAUDE-SETUP.md).

## The dashboard

```bash
vestige dashboard
```

A living WebGPU observatory of your memory at **http://localhost:3927/dashboard**: memories appear, link, strengthen, and fade in real time, 1000+ nodes at 60fps. It renders a deterministic 12-second loop of your store's life that you can export as an mp4 with one click, and mints a **brain print**, a signature seeded from your store's shape. Share artifacts are structure-only by design: your brain, never your memories.

## Vestige Pro

Everything above is free forever and never metered. **Pro ($19/month)** is managed, end-to-end encrypted continuity: your memory graph and accountability history (receipts, traces, memory PRs) following you across machines. XChaCha20-Poly1305 applied on your device, Argon2id over a passphrase only you know, ciphertext-only server. Zero-knowledge is the design: lose the passphrase and the data is unrecoverable, by anyone. Checkout opens shortly; watch [Releases](https://github.com/samvallad33/vestige/releases) for the announcement.

## Under the hood

| | |
|---|---|
| Engine | Rust 2024, ~145k lines, single 25MB binary, 2,000+ tests, clippy clean at `-D warnings` |
| Retrieval | Nomic Embed v1.5 (Matryoshka 768d→256d) + USearch HNSW + SQLite FTS5, optional Qwen3 reranker |
| Storage | SQLite, optional SQLCipher encryption ([docs/STORAGE.md](docs/STORAGE.md)) |
| Offline | Two model downloads on first run (130MB embedder, ~150MB reranker), then no network, ever |

## Go deeper

[Getting Started](docs/GETTING-STARTED.md) · [FAQ](docs/FAQ.md) · [The Science](docs/SCIENCE.md) · [Configuration](docs/CONFIGURATION.md) · [Storage](docs/STORAGE.md) · [Silent Rotation](https://github.com/samvallad33/vestige/tree/benchmark/silent-rotation/benchmarks/silent-rotation) · [Changelog](CHANGELOG.md)

---

If Vestige saves you from one repeated mistake, that is the whole point: **never solve the same problem twice.** If it earns a place in your setup, [a star](https://github.com/samvallad33/vestige) genuinely helps.

Built by [Sam](https://github.com/samvallad33). Licensed under [AGPL-3.0](LICENSE).
