# Vestige

[![MCP Toplist](https://mcptoplist.com/badge/io.github.samvallad33%2Fvestige.svg)](https://mcptoplist.com/server/io.github.samvallad33%2Fvestige)

Local-first long-term memory for AI agents, delivered over MCP. Vestige remembers your decisions, catches contradictions before they cost you, and traces a failure back to the older memory that actually caused it. One 25MB Rust binary. No cloud. Your data never leaves your machine.

[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/badge/tests-1550_passing-22c55e)](https://github.com/samvallad33/vestige/actions)
[![Binary](https://img.shields.io/badge/binary-25MB_single_file-informational)](https://github.com/samvallad33/vestige/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6)](LICENSE)

[What it is](#what-vestige-is) · [Install](#install) · [First interaction](#your-first-real-interaction) · [vs RAG](#how-it-differs-from-rag) · [Backward reach](#backward-reach-the-backfill-feature) · [Benchmark](#silent-rotation-a-reproducible-benchmark) · [Science](#the-science) · [Tools](#the-13-tools) · [Dashboard](#the-dashboard) · [Integrations](#works-with-every-agent) · [Pro](#vestige-pro) · [Docs](#go-deeper)

---

## What Vestige is

Hi, I'm [Sam](https://github.com/samvallad33). I built Vestige because my agents kept re-learning the same lessons. They would recommend a change I had already tested and rejected, re-derive a fix that was already written down, and treat every session as if the last one never happened.

Vestige is the memory layer that fixes that. It runs locally as an MCP server, so any MCP-capable agent (Claude Code, Claude Desktop, Codex, Cursor, and others) can write memories during a session and retrieve them later. Your data lives in a SQLite file on your own machine. After a one-time model download it works fully offline, with no API keys and no telemetry.

The part that makes it more than a note store: Vestige models memory on real cognitive science. It merges what is redundant, supersedes what is contradicted, keeps what you actually use, and lets unused memories fade. Most importantly, when a failure hits it can reach backward to the earlier decision that caused it, even when the cause and the symptom share no vocabulary. The cause never looks like the bug.

---

## Install

Three steps. You need Node.js installed (for the npm command) and nothing else.

### 1. Install the server

No Docker, no API key, no signup.

```bash
npm install -g vestige-mcp-server@latest
```

This installs the `vestige-mcp` command. Prebuilt binaries ship for macOS (Apple Silicon and Intel), Linux x86_64, and Windows x86_64, so there is no compile step.

### 2. Connect it to your agent

Vestige speaks [MCP](https://modelcontextprotocol.io), so it works with any MCP-capable agent. Every MCP client understands this config. Add it to your client's MCP settings:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

If you prefer the CLI, use the one-line shortcut for your agent:

| Agent | Setup |
|---|---|
| Claude Code | `claude mcp add vestige vestige-mcp -s user` |
| Codex | `codex mcp add vestige -- vestige-mcp` |
| Cursor / VS Code / Windsurf | add the JSON above to the editor's MCP settings, or see [docs/integrations/](docs/integrations/) |
| Cline / Continue / Zed / Goose | add the JSON above to that client's MCP config |
| Claude Desktop | [docs/CONFIGURATION.md#claude-desktop-macos](docs/CONFIGURATION.md#claude-desktop-macos) |

### 3. Verify

On first run, Vestige downloads its embedding model once (about 130MB). After that it never needs the network again. To confirm the server is healthy, open the dashboard:

```bash
vestige dashboard
```

Then visit **http://localhost:3927/dashboard**. If you see the graph, you are connected. For a fuller walkthrough see [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

---

## Your first real interaction

Memories go in as you work. The interesting behavior shows up when a new claim conflicts with something you already stored.

Say your agent recorded this earlier:

> We use Postgres for the primary datastore. Decided against MySQL for the JSONB support.

Later, someone tells the agent the opposite:

> Our primary datastore is MySQL.

When the agent tries to store that, Vestige does not silently append it. The engine returns a `claim_contradicts_memory` status and surfaces the older, conflicting memory, so the agent can resolve the conflict instead of quietly holding two incompatible facts.

The other command you will reach for is backfill. When something breaks, run:

```bash
vestige backfill --contrast
```

This walks backward from the failure to the earlier memory that most plausibly caused it, and shows you the contrast between what you believed then and what went wrong now. That backward reach is the feature the rest of this README builds up to.

---

## How it differs from RAG

RAG retrieves text that resembles your query. That is the right tool when the answer looks like the question. It is the wrong tool when the cause of a problem looks nothing like the symptom.

| | Plain RAG / vector search | Vestige |
|---|---|---|
| Retrieval basis | Text similarity to the query | Causal and temporal links, plus similarity |
| Finding a root cause | Cannot, because the cause does not resemble the bug | Reaches backward to the root-cause memory |
| Contradictions | Stored side by side, both returned | Detected and flagged (`claim_contradicts_memory`) |
| Redundant writes | Accumulate as duplicates | Merged on write via prediction-error gating |
| Unused memories | Persist at full weight | Fade over time (FSRS-6 spaced repetition) |
| Where it runs | Usually a cloud service | Local single binary, offline after setup |
| Your data | Leaves your machine | Never leaves your machine |

The distinction is not marketing. DeepMind proved that single-vector retrieval is mathematically incapable of representing certain relevance patterns ([arXiv:2508.21038](https://arxiv.org/abs/2508.21038), ICLR 2026). That theorem is about the limits of the vector-only approach. The measured gap on the task below is my own.

---

## Backward reach: the backfill feature

Most memory systems only look forward: you ask a question, they return similar text. Vestige also looks backward.

When a failure lands, the useful memory is rarely the one that resembles the error message. It is an older decision, made in different words, that set the failure up. A config choice from three weeks ago. A library pin. An assumption nobody wrote down as risky at the time.

Vestige implements **Retroactive Salience Backfill** (Zaki, Cai et al., *Nature* 2024, 637:145-155, [DOI 10.1038/s41586-024-08168-4](https://doi.org/10.1038/s41586-024-08168-4)). When a memory turns out to matter, the system reaches backward and raises the salience of the earlier memories that led to it, so the causal chain becomes retrievable even though the surface text never matched.

In practice you run `vestige backfill --contrast`. Vestige returns the earlier memory that most plausibly caused the current failure, alongside the contradiction between then and now. It finds the cause you would not have thought to search for.

---

## Silent Rotation: a reproducible benchmark

The claim above is testable, and the test ships with every transcript it produced.

**Silent Rotation** lives at [`benchmarks/silent-rotation/`](https://github.com/samvallad33/vestige/tree/benchmark/silent-rotation/benchmarks/silent-rotation). Three coding agents fix one failing end-to-end test in a TypeScript monorepo. The fix needs the currently live signing key id, which is randomized per trial from a 50-key keyring and appears in no file the agents can read. It exists only in the memory layer.

Reproduce the central result in two seconds. Python standard library only, no API keys, no network:

```bash
git clone -b benchmark/silent-rotation --depth 1 https://github.com/samvallad33/vestige.git
cd vestige/benchmarks/silent-rotation
python3 tests/bm25_baseline.py results/runA-trial-1/corpus-export.json --no-dense
```

**What it measures.** A fleet either converges on the correct key, converges on a planted decoy, or splits and fails to merge. The second outcome is the dangerous one: tests pass, the merge is clean, and production breaks.

**The numbers.** 6 models, 25 trials, 246 published agent transcripts.

| Arm | Converged correct | Converged **wrong** | Split |
|---|---|---|---|
| No memory | 0/25 | **21/25** | 4/25 |
| Dense cosine RAG | 4/23 | **12/23** | 7/23 |
| Vestige | 20/23 | **0/23** | 3/23 |

Two separate claims, kept separate on purpose:

1. **The theorem (DeepMind).** Single-vector retrieval is mathematically incapable of these relevance gaps ([arXiv:2508.21038](https://arxiv.org/abs/2508.21038), ICLR 2026). This is a fundamental limit of vector search.
2. **The measurement (mine).** On the verbatim queries the agents actually typed, the causal memory ranks 7th of 8 under both dense cosine *and* BM25, while the decoy ranks 1st.

The caveats are published alongside the results, including the trials where a plain cosine baseline ties Vestige and the trial Vestige loses.

---

## The science

Every mechanism below is a cited result, implemented in Rust, running locally. None of it calls a cloud model to sound smart. Full write-up in [docs/SCIENCE.md](docs/SCIENCE.md).

| Mechanism | What it does | Source |
|---|---|---|
| Prediction-Error Gating | Stores only what is novel: merges redundant, supersedes contradictory | Hippocampal novelty gating |
| FSRS-6 spaced repetition | 21-parameter schedule so used memories persist and unused ones fade | Modern spaced-repetition research |
| Retroactive Salience Backfill | Reaches backward to a failure's root-cause memory | Zaki, Cai et al. 2024, *Nature* 637:145-155, [10.1038/s41586-024-08168-4](https://doi.org/10.1038/s41586-024-08168-4) |
| Synaptic Tagging | Marks memories for later consolidation | Frey & Morris 1997, [10.1038/385533a0](https://doi.org/10.1038/385533a0) |
| Spreading Activation | Retrieving one memory activates related ones through the graph | Collins & Loftus 1975, [10.1037/0033-295X.82.6.407](https://doi.org/10.1037/0033-295X.82.6.407) |
| Dual-Strength | Separates how well something is stored from how easily it is retrieved | Bjork & Bjork 1992 |
| Memory Dreaming | Sleep-like consolidation that replays and synthesizes memories | Sleep consolidation and replay |
| Active Forgetting | Top-down inhibition that suppresses a memory, cascades to neighbors, reversible for 24 hours | Anderson 2025, Davis 2020 |

---

## The 13 tools

Vestige exposes exactly 13 MCP tools. Your agent calls them; you rarely call them by hand.

| Tool | Purpose |
|---|---|
| `recall` | Retrieve memories relevant to the current context |
| `backfill` | Reach backward from a failure to its root-cause memory |
| `smart_ingest` | Store a fact, with gating for novelty and contradiction |
| `memory` | Read, inspect, promote, or demote individual memories |
| `graph` | Explore the memory graph and its links |
| `maintain` | Run consolidation and lifecycle maintenance |
| `dedup` | Find and merge duplicate memories |
| `suppress` | Actively forget a memory (reversible for 24h) |
| `memory_status` | Report health, counts, and model readiness |
| `codebase` | Index and query codebase-scoped memory |
| `intention` | Track goals and open intentions across sessions |
| `source_sync` | Sync memories from external connected sources |
| `session_start` | Prime the agent with relevant context at session start |

---

## The dashboard

```bash
vestige dashboard
```

Open **http://localhost:3927/dashboard** to watch your memory as a live 3D graph.

It is built with SvelteKit 2 and Svelte 5, rendering with WebGPU and Three.js with bloom, driven by a live WebSocket feed, holding 1000+ nodes at 60fps. Memories appear, link, strengthen, and fade in real time as your agent works. It installs as a PWA if you want it as a standalone app.

---

## Works with every agent

Vestige is a standard MCP server, so it works with any MCP-capable client. The universal config is all most agents need:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

| Client | Setup |
|---|---|
| Claude Code | `claude mcp add vestige vestige-mcp -s user` |
| Codex | `codex mcp add vestige -- vestige-mcp` |
| Cursor | [docs/integrations/cursor.md](docs/integrations/cursor.md) |
| VS Code | [docs/integrations/vscode.md](docs/integrations/vscode.md) |
| Windsurf | [docs/integrations/windsurf.md](docs/integrations/windsurf.md) |
| Claude Desktop | [docs/CONFIGURATION.md#claude-desktop-macos](docs/CONFIGURATION.md#claude-desktop-macos) |
| Cline / Continue / Zed / Goose | add the universal config above |

Full configuration reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md). Intel Mac notes: [docs/INSTALL-INTEL-MAC.md](docs/INSTALL-INTEL-MAC.md).

---

## Optional: make the agent use memory automatically

By default your agent calls the tools when it decides to. If you want memory to be a standing habit (recall at the start of a task, save durable facts as they land), give the agent a short protocol.

- General agent memory protocol: [docs/AGENT-MEMORY-PROTOCOL.md](docs/AGENT-MEMORY-PROTOCOL.md)
- Claude-specific setup and templates: [docs/CLAUDE-SETUP.md](docs/CLAUDE-SETUP.md)

This is opt-in. Vestige works fine with no protocol at all.

---

## Vestige Pro

Everything above is free forever and never metered. The engine runs on your machine, with no account, no quota, and no upsell inside the product.

Vestige Pro is for when that memory needs to follow you. It is managed, end-to-end encrypted continuity of your memory graph and your accountability history (Black Box traces, receipts, memory PRs) across every machine you work on. You record a decision on the laptop, and the agent on the desktop already knows it.

| | Detail |
|---|---|
| Price | $19/month |
| What syncs | Your memory graph plus your accountability history |
| Encryption | XChaCha20-Poly1305, applied on your machine before anything is uploaded |
| Key derivation | Argon2id over a passphrase you choose |
| What the server holds | Ciphertext only |

Zero-knowledge is the design, not a setting. You pick one passphrase, you use the same one on every device, and it never leaves your machine. The server stores bytes it cannot read, and the client refuses to sync anything in plaintext. If you lose that passphrase, the encrypted data is unrecoverable, by me and by anyone else. That is the property you are paying for, not a gap in it.

**Availability.** Checkout is not open yet, so there is nothing to buy today and no payment link here pretending otherwise. The client half already ships in this release, which is why `vestige sync --cloud` exists and tells you what it needs. Subscriptions open shortly. To catch the announcement, watch [Releases](https://github.com/samvallad33/vestige/releases) or follow [Discussions](https://github.com/samvallad33/vestige/discussions).

---

## Under the hood

Vestige is a single Rust binary. No sidecar services, no external database, no cloud dependency.

| Component | Detail |
|---|---|
| Language | Rust 2024 edition, about 96,000 lines |
| Distribution | Single 25MB binary, prebuilt for all platforms |
| Embeddings | Nomic Embed Text v1.5 (768d reduced to 256d via Matryoshka, 8192-token context) |
| Reranker | Qwen3 reranker, optional |
| Vector search | USearch HNSW |
| Storage | SQLite with FTS5, optional SQLCipher encryption |
| First run | Downloads about 130MB embedding model once, then fully offline forever |
| Platforms | macOS (ARM + Intel), Linux x86_64, Windows x86_64, all prebuilt |
| Quality | 1,550 tests passing, clippy clean with `-D warnings` |

Storage internals and encryption: [docs/STORAGE.md](docs/STORAGE.md).

---

## Go deeper

| Doc | What's in it |
|---|---|
| [Getting Started](docs/GETTING-STARTED.md) | Full first-run walkthrough |
| [FAQ](docs/FAQ.md) | Common questions |
| [The Science](docs/SCIENCE.md) | Every mechanism with its citation |
| [Configuration](docs/CONFIGURATION.md) | All options and per-agent setup |
| [Storage](docs/STORAGE.md) | Storage format and encryption |
| [Agent Memory Protocol](docs/AGENT-MEMORY-PROTOCOL.md) | Teaching an agent to use memory automatically |
| [Intel Mac install](docs/INSTALL-INTEL-MAC.md) | Notes for older Macs |
| [Silent Rotation](https://github.com/samvallad33/vestige/tree/benchmark/silent-rotation/benchmarks/silent-rotation) | The reproducible benchmark |
| [Changelog](CHANGELOG.md) | Release history |

---

If Vestige saves you from one repeated mistake, that is the whole point: never solve the same problem twice. If it earns a place in your setup, [star it on GitHub](https://github.com/samvallad33/vestige). It genuinely helps me keep building.

Built by [Sam](https://github.com/samvallad33). Licensed under [AGPL-3.0](LICENSE).
