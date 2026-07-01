<div align="center">

<h1>Vestige</h1>

### Your bug was born days before it crashed. You just can't remember where.

<em>Vestige is a local-first memory for AI agents that reaches <b>backward through time</b> to find the quiet change that caused today's failure: the cause that looks nothing like the bug. One 23MB Rust binary. No cloud. Your data never leaves your machine.</em>

[![GitHub stars](https://img.shields.io/github/stars/samvallad33/vestige?style=for-the-badge&logo=github&color=8b5cf6)](https://github.com/samvallad33/vestige/stargazers)
[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?style=for-the-badge&color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/badge/tests-1550_passing-22c55e?style=for-the-badge)](https://github.com/samvallad33/vestige/actions)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6?style=for-the-badge)](LICENSE)

[**⚡ Quick Start**](#-get-it-running-in-60-seconds) · [**🧠 The Idea**](#-why-i-built-this) · [**🔬 The Science**](#-this-is-real-neuroscience-not-a-metaphor) · [**🛠 13 Tools**](#-13-tools-one-brain) · [**📊 Dashboard**](#-watch-your-ai-think-in-3d)

</div>

---

## 👋 Why I built this

Hi, I'm [Sam](https://github.com/samvallad33). I built Vestige from a tiny apartment in Chicago because I kept losing days to the same thing, and I bet you have too.

Production breaks. You start hunting. And the cause is almost never *near* the error. It's some quiet change you made days ago that looks **nothing** like the crash it eventually caused. A flipped env var. A swapped service. A config tweak you'd already forgotten.

Here's the part that took me a while to see: **every AI memory tool is built on vector search, and vector search hunts for what *looks like* your problem.** But a root cause never looks like the bug it creates. So they all search the goal line, while the real failure was a quiet midfield turnover fifteen minutes earlier.

I wanted a memory that traces the match *backward.*

So that's what Vestige is. Everyone else built a memory that **remembers**. I tried to build the first one that **realizes**: it gates what's worth keeping, lets the noise fade like your own memory does, and when a failure hits, it reaches back through time to the change that actually caused it.

It's one Rust binary. It runs entirely on your machine. It never phones home. And there's a 60-second start right below.

> 🎙️ **Want the 60-second version?** Run the [Postdict demo](demo/README.md): one command, a throwaway DB, and you watch Vestige reach backward through time to a root cause a vector search can't find. It's the clearest way to *get* why this matters.

---

## ⚡ Get it running in 60 seconds

**Step 1 — install (one binary, no Docker, no API key, no signup):**

```bash
npm install -g vestige-mcp-server@latest
```

**Step 2 — connect it to your agent.** Vestige speaks [MCP](https://modelcontextprotocol.io), so it works with *any* AI agent. The universal config (works everywhere):

```json
{ "mcpServers": { "vestige": { "command": "vestige-mcp" } } }
```

Drop that into your agent's MCP config file. Or use the one-line shortcut for your agent:

```bash
# Cursor / Windsurf / VS Code      → add the JSON above to ~/.cursor/mcp.json (or the editor's MCP settings)
# Claude Code                      → claude mcp add vestige vestige-mcp -s user
# Codex                            → codex mcp add vestige -- vestige-mcp
# Cline / Continue / Zed / Goose   → add the JSON above to that client's MCP config
```

**Step 3 — confirm it's working:**

```bash
vestige-mcp --version     # prints the installed version
vestige stats             # prints your memory count (0 on a fresh install)
```

That's the whole install. Per-agent guides (Cursor, VS Code, Windsurf, JetBrains, Xcode, OpenCode, Codex, Claude Desktop) are [here ↓](#-works-with-every-ai-agent).

Now talk to your agent like it has a memory, because now it does:

```
You:  "Remember: we always disable SimSIMD on release builds, it breaks old x86 CPUs."
        ...days later, fresh session, zero context...
You:  "Should I enable SimSIMD for the release?"
AI:   ⚠️ Hold on, this contradicts a decision you stored: you chose to DISABLE it
        because it breaks old x86 CPUs.
```

That last line isn't me being cute. It's a real status the engine returns, called `claim_contradicts_memory`. Most memory tools would have happily handed you the wrong answer. Vestige tells you when you're about to walk back into a mistake you already learned from.

And the headline feature, the one nothing else does, is one command:

```bash
vestige backfill --contrast
```

When a failure is in your memory, this reaches *backward through time* and finds the quiet earlier change that caused it (the one a vector search ranks poorly because it shares no words with the error). It shows you, side by side, what similarity search returns versus the real cause. [More on the backward reach ↓](#-the-thing-nothing-else-does-memory-with-hindsight)

*(Works with Codex, Cursor, VS Code, Claude Desktop, Windsurf, JetBrains, Zed: anything that speaks MCP. [Full setup is here ↓](#-works-in-every-editor-you-use).)*

---

## 🧠 It's not RAG with a nicer haircut

RAG is a bucket: throw everything in, hope nearest-neighbor finds it later. Vestige behaves more like an actual memory: it decides what's worth keeping, forgets what isn't, and reasons across what's left.

|  | 🪣 RAG / Vector Store | 🧠 Vestige |
|---|---|---|
| **What it stores** | Everything you hand it | Only what's **surprising or new** (the rest gets merged or skipped) |
| **What it forgets** | Nothing; it just bloats | Unused memories **fade** on a real forgetting curve, so your context stays lean |
| **Finding a root cause** | Can't, because the cause isn't *similar* to the bug | **Reaches backward in time** to the change that caused it (the whole point ↓) |
| **Catching contradictions** | Silent; serves the stale answer with a straight face | Tells you: *"this contradicts what you decided"* |
| **Duplicates** | You clean them up by hand | Self-heals: *"likes dark mode"* + *"prefers dark themes"* quietly become one |
| **Forgetting on demand** | DELETE and it's gone | **`suppress`** gently inhibits a memory (and its neighbors), reversible for 24h |
| **Where it lives** | Usually someone else's cloud | **Your machine. One binary. No telemetry.** |

---

## 🔥 The thing nothing else does: memory with hindsight

This is the part I'm proudest of, and it's worth one honest paragraph.

A bug shows up today. The cause was a quiet decision from three weeks ago, like a changed env var or a swapped service. That cause **shares no words with the error it created.** A vector search will never connect them, because it only knows how to find things that *look alike*, and this is a case where the cause and the symptom look nothing alike. This isn't a tuning problem; in 2026 Google DeepMind published a proof ([arXiv:2508.21038](https://arxiv.org/abs/2508.21038), ICLR 2026) that single-vector retrieval is *mathematically* incapable of bridging gaps like this.

So Vestige doesn't do it with similarity. Its **Retroactive Salience Backfill** (ported from **Zaki/Cai et al., 2024, *Nature* 637:145–155** ([DOI](https://doi.org/10.1038/s41586-024-08168-4)), on how the brain links a shock to the quiet memory that caused it) reaches *backward through time* and promotes the dormant memory that's **causally upstream**: it shares an *entity* (the same file, env var, or service), not the same words.

I also built a benchmark to keep myself honest about it. Every pure vector retriever scored **0% recall@1** on the causal-gap task; Vestige scored **60%**. (To be precise: the impossibility is DeepMind's *theorem*; the 0%-vs-60% is *my measurement*. Two different claims, and I keep them separate.)

```bash
vestige backfill --contrast      # show the root cause a vector search would have missed
```

The nice part: it compounds. Every failure your agent records makes the *next* session diagnose faster (run two is smarter than run one), and it happens automatically during consolidation, so you don't have to babysit it.

All of this shipped in **v2.2.0**, along with a 34→13 tool consolidation and a rebuilt retrieval engine. [Full release notes →](https://github.com/samvallad33/vestige/releases/tag/v2.2.0)

---

## 🔬 This is real neuroscience, not a metaphor

I get skeptical when projects wave the word "neuroscience" around, so here's my receipt: every mechanism below is a real, cited paper, implemented in Rust, running locally on your machine. None of it phones a model in the cloud to sound smart.

| Mechanism | What it does for you | Grounded in |
|---|---|---|
| **Prediction-Error Gating** | Redundant info gets merged, contradictory gets superseded, only the novel gets stored | The hippocampal novelty signal |
| **FSRS-6 Spaced Repetition** | 21 parameters of the mathematics of forgetting, so used memories stay and unused ones fade | Modern spaced-repetition research |
| **Retroactive Salience Backfill** | Backward causal reach to the root cause of a failure | Zaki/Cai et al. 2024, *Nature* 637:145–155 |
| **Synaptic Tagging** | A memory that looked trivial this morning can be tagged critical tonight | [Frey & Morris 1997](https://doi.org/10.1038/385533a0) |
| **Spreading Activation** | Search "auth bug," surface last week's JWT update, because memory is a graph, not a list | [Collins & Loftus 1975](https://doi.org/10.1037/0033-295X.82.6.407) |
| **Dual-Strength Model** | Storage strength vs. retrieval strength, so deeply stored ≠ instantly recalled, just like you | [Bjork & Bjork 1992](https://doi.org/10.1016/S0079-7421(08)60016-9) |
| **Memory Dreaming** | Sleep-like consolidation: replays, connects, synthesizes insights to a graph | Active-dreaming consolidation |
| **Active Forgetting (`suppress`)** | Top-down inhibition that *compounds* and cascades to neighbors, reversible for 24h | [Anderson 2025](https://www.nature.com/articles/s41583-025-00929-y) · [Davis 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7477079/) |

[**Read the full science doc →**](docs/SCIENCE.md). Every feature, every paper.

---

## 🛠 13 tools, one brain

v2.2.0 consolidated a sprawling 34-tool surface into **13 sharp ones** your agent actually reaches for. Old names still work as hidden aliases, so nothing breaks.

| Tool | What it does |
|---|---|
| 🔍 `recall` | The retrieval engine. Folds search + deep reasoning + contradiction detection into one call. F32 embeddings, Reciprocal Rank Fusion, claim-vs-memory checks. |
| 🧠 `backfill` | **Memory with hindsight.** Backward causal reach to a failure's root cause (Cai 2024). |
| 💾 `smart_ingest` | Stores with CREATE / UPDATE / SUPERSEDE via Prediction-Error Gating. Batch session-end saves. |
| 🗂 `memory` | Get, edit, promote 👍, demote 👎, check state, purge content + embeddings. |
| 🧩 `graph` | Reasoning chains, associations, bridges, predictions, force-directed export. |
| 🌙 `maintain` | Consolidate, dream, GC, importance-score, backup, export, restore. One maintenance verb. |
| 🧹 `dedup` | Self-healing duplicate detection + merge (8 old tools → 1). |
| 🚫 `suppress` | Top-down active forgetting that compounds, cascades, and is reversible for 24h. The memory is *inhibited, not erased.* |
| 📟 `memory_status` | Health + stats + trends + recommendations in one packet. |
| 🧬 `codebase` · `intention` · `source_sync` · `session_start` | Per-project code memory · "remind me when X" · external-source connectors · one-call session init. |

---

## 📊 Watch your AI think in 3D

```bash
vestige dashboard      # → http://localhost:3927/dashboard
```

Every memory is a glowing node in a real-time, force-directed 3D graph. Connections form as you work. Nodes **pulse** when accessed, **burst** on creation, **fade** on decay. Kick off a consolidation and the whole graph slides into **purple dream mode**, replaying memories that light up in sequence.

Built with SvelteKit 2 · Svelte 5 · Three.js · WebGL bloom · live WebSocket events. 1000+ nodes at 60fps. Installable as a PWA.

---

## 🧩 Works with every AI agent

Vestige speaks MCP, so **any agent that can register an MCP server can use it.** Not a plugin for one tool, the memory layer underneath all of them. The universal config works everywhere:

```json
{ "mcpServers": { "vestige": { "command": "vestige-mcp" } } }
```

| Agent | Setup |
|---|---|
| **Cursor** | add the JSON above to `~/.cursor/mcp.json` · [guide →](docs/integrations/cursor.md) |
| **Windsurf** | [guide →](docs/integrations/windsurf.md) |
| **VS Code (Copilot)** | [guide →](docs/integrations/vscode.md) |
| **Cline / Continue / Zed / Goose** | add the universal JSON to that client's MCP config |
| **Claude Code** | `claude mcp add vestige vestige-mcp -s user` |
| **Codex** | `codex mcp add vestige -- vestige-mcp` |
| **JetBrains · Xcode · OpenCode** | [integration guides →](docs/integrations/) |
| **Claude Desktop** | [2-minute setup →](docs/CONFIGURATION.md#claude-desktop-macos) |

<details>
<summary><b>Other install methods (Intel Mac, Windows, build-from-source)</b></summary>

**Update an existing install:**
```bash
vestige update                          # binaries only
vestige update --sandwich-companion     # also refresh optional Claude Code companion files
```

**macOS (Intel):** Microsoft is dropping x86_64 macOS ONNX Runtime prebuilts after v1.23.0, so the Intel Mac build links dynamically against a Homebrew ONNX Runtime:
```bash
brew install onnxruntime
npm install -g vestige-mcp-server@latest
echo 'export ORT_DYLIB_PATH="'"$(brew --prefix onnxruntime)"'/lib/libonnxruntime.dylib"' >> ~/.zshrc && source ~/.zshrc
claude mcp add vestige vestige-mcp -s user
```
Full guide: [`docs/INSTALL-INTEL-MAC.md`](docs/INSTALL-INTEL-MAC.md).

**Windows + Claude Desktop:** quit Claude Desktop from the tray, then in PowerShell:
```powershell
npm install -g vestige-mcp-server@latest
vestige-mcp --version
```
Point `%APPDATA%\Claude\claude_desktop_config.json` at it:
```json
{ "mcpServers": { "vestige": { "command": "vestige-mcp" } } }
```
If it can't find the command, run `where vestige-mcp` and use the exact `.cmd` path.

**Build from source (Rust 1.91+):**
```bash
git clone https://github.com/samvallad33/vestige && cd vestige
cargo build --release -p vestige-mcp
# Apple Silicon GPU: --features metal   ·   NVIDIA: --features qwen3-embeddings,cuda
```
</details>

---

## 🚀 Make your AI use memory automatically

Registering the server exposes the tools; a short instruction tells the agent *when* to call them. Drop in the protocol and your agent saves and recalls on its own:

| You say | Vestige does |
|---|---|
| *"Remember this"* | Saves immediately |
| *"I always..."* / *"I prefer..."* | Saves as a durable preference |
| *"Remind me when..."* | Creates a future trigger (`intention`) |
| *"This is important"* | Saves **and** promotes it |

[Agent memory protocol →](docs/AGENT-MEMORY-PROTOCOL.md) · [Claude Code template →](docs/CLAUDE-SETUP.md)

---

## 🏗 Under the hood

```
┌──────────────────────────────────────────────────────────┐
│  SvelteKit Dashboard / Three.js 3D graph / WebGL bloom    │
├──────────────────────────────────────────────────────────┤
│  Axum HTTP + WebSocket (:3927) / REST + live event stream │
├──────────────────────────────────────────────────────────┤
│  MCP Server (stdio JSON-RPC) / 13 tools · 30 modules      │
├──────────────────────────────────────────────────────────┤
│  Cognitive Engine                                          │
│   FSRS-6 · Spreading Activation · Prediction-Error Gating │
│   Retroactive Salience Backfill · Synaptic Tagging        │
│   Memory Dreamer · Hippocampal Index · Active Forgetting  │
├──────────────────────────────────────────────────────────┤
│  Storage: SQLite + FTS5 · USearch HNSW · Nomic Embed v1.5 │
│   Optional: Qwen3 reranker · SQLCipher · Metal/CUDA       │
└──────────────────────────────────────────────────────────┘
```

| | |
|---|---|
| **Language** | Rust 2024 (MSRV 1.91), **92,000+ lines** |
| **Binary** | ~23MB, single file |
| **Embeddings** | Nomic Embed Text v1.5 (768d→256d Matryoshka, 8192 ctx); Qwen3 optional |
| **Vector search** | USearch HNSW (≈20× faster than FAISS) |
| **Storage** | SQLite + FTS5, optional SQLCipher encryption |
| **Tests** | **1,550 passing** · clippy `-D warnings` clean |
| **First run** | Downloads ~130MB embedding model once, then **fully offline forever** |
| **Platforms** | macOS (ARM + Intel) · Linux x86_64 · Windows x86_64. All prebuilt |

---

## 📚 Go deeper

| | |
|---|---|
| [**FAQ**](docs/FAQ.md) | 30+ real questions answered |
| [**The Science**](docs/SCIENCE.md) | Every feature, every paper |
| [**Storage Modes**](docs/STORAGE.md) | Global · per-project · multi-instance |
| [**Configuration**](docs/CONFIGURATION.md) | CLI, env vars, every knob |
| [**Changelog**](CHANGELOG.md) | The full story, version by version |

---

<div align="center">

### If your agent should remember what you taught it yesterday, star it. ⭐

<sub><b>92,000+ lines of Rust · 13 tools · 30 cognitive modules · 130 years of memory research · one 23MB binary that never phones home.</b></sub>

<sub>Built by <a href="https://github.com/samvallad33">@samvallad33</a> · AGPL-3.0 · 100% local, 100% yours</sub>

</div>
