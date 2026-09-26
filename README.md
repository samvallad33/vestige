<p align="center">
  <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-logo.png" alt="Vestige" width="620">
</p>

# Vestige

**Cognitive Deterministic Memory Transaction-Security OS for Agentic AI.**

Your agent can think anything. Vestige decides what it is allowed to do.

It remembers every decision your project ever made, reaches backward through time to find the quiet choice behind today's failure, and blocks the actions your agent should never take. Receipt, one-use permit, effect. Or STOP with zero effects. Local. Encrypted. No cloud, no telemetry.

[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/github/actions/workflow/status/samvallad33/vestige/ci.yml?branch=main&label=CI)](https://github.com/samvallad33/vestige/actions)
[![Binary](https://img.shields.io/badge/platforms-5_release_targets-informational)](https://github.com/samvallad33/vestige/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6)](LICENSE)

[Install](#install) · [Why not RAG](#why-not-just-rag) · [The Live Gate](#founding-operator) · [Continuity](#managed-continuity) · [Benchmark](#the-receipts-silent-rotation) · [Science](#the-science) · [Docs](#go-deeper)

<a id="getting-started"></a>
## The cause never looks like the bug

Agents re-learn the same lessons. They recommend a change you already tested and rejected, re-derive a fix that was already written down, and treat every session as if the last one never happened. Vestige is the deterministic memory security OS that ends that. Any MCP-capable agent writes memories as you work and retrieves them later: redundant memories merge, contradicted ones are flagged, unused ones fade, and when a failure hits, Vestige reaches **backward** to the decision that set it up.

<p align="center">
  <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">
    <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/black-box-cause.gif" alt="Vestige Black Box: a SIGSEGV on startup traced back to a version pin set 23 days earlier, with the receipt" width="100%">
  </a>
</p>

<p align="center"><sub><b>A labeled fixture store, a real run.</b> A SIGSEGV on startup in an arm64 container, and the version pin set 23 days earlier that shares zero words with the failure. Similarity ranked the pin fourth. Backfill ranked it first. It names the suspects and never calls the verdict. <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">Watch the 58 second walk</a>.</sub></p>

## Install

You need Node.js. No Docker, no signup, no compile step. Prebuilt for macOS ARM and Intel, Linux x86_64 and arm64, Windows x86_64.

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

Verify it: `vestige dashboard`, then open **http://localhost:3927/dashboard**. First run downloads a 130MB embedding model once; after that Vestige is fully offline, forever. Full walkthrough: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

## Why not just RAG?

RAG retrieves text that resembles the query. That works when the answer looks like the question. It fails when the cause of a problem looks nothing like the symptom.

| | Vector search | Vestige |
|---|---|---|
| Retrieval basis | Similarity to the query | Causal and temporal links, plus similarity |
| Root cause of a failure | Cannot. The cause does not resemble the bug | `vestige backfill --contrast` reaches backward to it |
| Contradictions | Both stored, both returned | Detected and flagged |
| Redundant writes | Accumulate | Merged on write |
| Unused memories | Persist at full weight | Fade (FSRS-6 spaced repetition) |
| Your data | Usually a cloud service | Never leaves your machine |

The backward reach implements Retroactive Salience Backfill (Zaki, Cai et al., *Nature* 2024): when a memory turns out to matter, the earlier memories that led to it become retrievable too. Every backfill result ships with a receipt naming the exact evidence path. DeepMind separately proved single-vector retrieval is mathematically incapable of certain relevance patterns ([arXiv:2508.21038](https://arxiv.org/abs/2508.21038)).

## 🛡️ Founding Operator

The fail-closed authority kernel for AI agents. Your agent can think anything. Operator decides what it is allowed to do.

Receipt, then a one-use permit, then the effect. Or STOP with zero effects. Thinking is not authority.

**See it stop a real agent: [▶ THE LIVE GATE (1:44)](https://github.com/samvallad33/vestige/releases/tag/launch-night-live-gate-20260914)**, recorded in one take. An agent tries to permanently purge its own memory trail. Operator refuses the effect, opens a Memory PR, and waits for a human.

- **$49 first month, first 100 people. $149/mo after.**
- Includes Managed Continuity.
- Login and checkout: **https://vestige-pro-production.fly.dev/account**

**What Operator buys, the daily ritual surface:**

- **Taste Lock** — the moment your agent commits to something consequential, it is locked and receipted before it can drift.
- **The Seven** — seven signed receipts of the last consequential writes, laid out like polaroids. What your agent actually did today, at a glance.
- **Almost-Forgot** — the three dim memories that are still load-bearing but fading, surfaced before they fail.
- **Night Letter** — a signed letter from your store at night: what changed, what contradicted, what decayed. Not a merge log. A letter.
- **Facepalm Backfill** — it broke again? Walk backward from the failure to the quiet decision that set it up, evidence path attached.
- **Canary Bite** — a secret about to leave the store gets eaten before it ships.

**Investigator ($79/mo)** is the instruments tier: deep backfill passes, evidence packets, retrieval tribunals, the full forensics surface for postmortems and audits. For engineers who debug with receipts.

<a id="managed-continuity"></a>
<a id="vestige-pro"></a>
## 🔄 Managed Continuity

Your agent's memory survives crashes, new machines, and reinstalls. Decisions, receipts, traces, and memory PRs follow you everywhere.

End to end encrypted: XChaCha20-Poly1305 applied on your device, Argon2id over a passphrase only you know, ciphertext-only server. Lose the passphrase and the data is unrecoverable, by anyone.

```bash
# 1. Full local backup
vestige backup ~/vestige-backup.db

# 2. Encrypted archive, drop it in iCloud, Dropbox, Syncthing, or Git
vestige sync ~/vestige-archive.vportable

# 3. Restore on any machine
vestige --data-dir ~/new-machine-store sync ~/vestige-archive.vportable
```

**$19/mo.** Subscribe at **https://vestige-pro-production.fly.dev/account**

## The receipts: Silent Rotation

The claim is testable, and the test ships with all 246 agent transcripts it produced. Three coding agents fix one failing e2e test; the fix needs a signing key id that exists only in the memory layer. The dangerous outcome is converging on a planted decoy: tests pass, the merge is clean, production breaks.

| Arm (6 models, 25 trials) | Converged correct | Converged wrong | Split |
|---|---|---|---|
| No memory | 0/25 | **21/25** | 4/25 |
| Dense cosine RAG | 4/23 | **12/23** | 7/23 |
| Vestige | 20/23 | **0/23** | 3/23 |

Reproduce the central measurement in two seconds:

```bash
git clone -b benchmark/silent-rotation --depth 1 https://github.com/samvallad33/vestige.git
cd vestige/benchmarks/silent-rotation
python3 tests/bm25_baseline.py results/runA-trial-1/corpus-export.json --no-dense
```

Caveats are published alongside the results, including the trials a plain cosine baseline ties.

## The science

Every mechanism is a cited result, implemented in Rust, running locally. Full write-up: [docs/SCIENCE.md](docs/SCIENCE.md).

| Mechanism | What it does | Source |
|---|---|---|
| Prediction-Error Gating | Stores only the novel; merges redundant | Hippocampal novelty gating |
| FSRS-6 spaced repetition | Used memories persist, unused ones fade | Modern spaced-repetition research |
| Retroactive Salience Backfill | Reaches backward to a failure's root cause | Zaki, Cai et al. 2024, *Nature* |
| Synaptic Tagging | Marks memories for later consolidation | Frey & Morris 1997 |
| Spreading Activation | One retrieval activates related memories | Collins & Loftus 1975 |
| Dual-Strength | Storage vs retrieval strength, tracked separately | Bjork & Bjork 1992 |
| Memory Dreaming | Sleep-like replay and synthesis | Sleep consolidation research |
| Active Forgetting | Reversible top-down suppression | Anderson 2025, Davis 2020 |

## The tools

Your agent calls these; you rarely do.

| Tool | Purpose |
|---|---|
| `recall` | Retrieve memories relevant to the current context |
| `smart_ingest` | Store a fact, gated for novelty and contradiction |
| `backfill` | Reach backward from a failure to its candidate cause |
| `receipt` | Inspect retrieval receipts and evidence replay |
| `project` | Project durable decisions into CLAUDE.md or MEMORY.md |
| `memory` · `graph` · `intention` | Inspect, promote, explore, track goals |
| `maintain` · `dedup` · `suppress` | Consolidation, merge, bounded suppression |

Full contracts: [docs/TOOL-CONTRACTS.md](docs/TOOL-CONTRACTS.md) · Hygiene and standing habits: [docs/MEMORY_HYGIENE.md](docs/MEMORY_HYGIENE.md)

## The dashboard

```bash
vestige dashboard
```

A living observatory of your memory at **http://localhost:3927/dashboard**: memories appear, link, strengthen, and fade in real time, 1000+ nodes at 60fps. It renders a deterministic 12-second loop of your store's life that you can export as an mp4 with one click. Share artifacts are structure-only by design: your brain, never your memories.

## Under the hood

| | |
|---|---|
| Engine | Rust 2024, ~145k lines, single 25MB binary, 2,000+ tests, clippy clean at `-D warnings` |
| Retrieval | Nomic Embed v1.5 (Matryoshka 768d→256d) + USearch HNSW + SQLite FTS5, optional Qwen3 reranker |
| Storage | SQLite, optional SQLCipher encryption ([docs/STORAGE.md](docs/STORAGE.md)) |
| Offline | Two model downloads on first run, then no network, ever |

## Go deeper

[Getting Started](docs/GETTING-STARTED.md) · [FAQ](docs/FAQ.md) · [The Science](docs/SCIENCE.md) · [Configuration](docs/CONFIGURATION.md) · [Storage](docs/STORAGE.md) · [Tool contracts](docs/TOOL-CONTRACTS.md) · [Changelog](CHANGELOG.md)

## License

AGPL-3.0. The hosted Continuity and Operator services are separate proprietary products.
