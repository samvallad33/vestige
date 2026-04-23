<div align="center">

# Vestige

### The cognitive engine that gives AI agents a brain.

[![GitHub stars](https://img.shields.io/github/stars/samvallad33/vestige?style=social)](https://github.com/samvallad33/vestige)
[![Release](https://img.shields.io/github/v/release/samvallad33/vestige)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/badge/tests-1284%20passing-brightgreen)](https://github.com/samvallad33/vestige/actions)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green)](https://modelcontextprotocol.io)

**Your Agent forgets everything between sessions. Vestige fixes that.**

Built on 130 years of memory research — FSRS-6 spaced repetition, prediction error gating, synaptic tagging, spreading activation, memory dreaming — all running in a single Rust binary with a 3D neural visualization dashboard. 100% local. Zero cloud.

[Quick Start](#quick-start) | [Dashboard](#-3d-memory-dashboard) | [How It Works](#-the-cognitive-science-stack) | [Tools](#-24-mcp-tools) | [Docs](docs/)

</div>

---

## What's New in v2.0.6 "Composer"

v2.0.6 is a polish release that makes the existing cognitive stack finally *feel* alive in the dashboard and stays out of your way on the prompt side.

- **Six live graph reactions, not one** — `MemorySuppressed`, `MemoryUnsuppressed`, `Rac1CascadeSwept`, `Connected`, `ConsolidationStarted`, and `ImportanceScored` now light the 3D graph in real time. v2.0.5 shipped `suppress` but the graph was silent when you called it; consolidation and importance scoring have been silent since v2.0.0. No longer.
- **Intentions page actually works** — fixes a long-standing bug where every intention rendered as "normal priority" (type/schema drift between backend and frontend) and context/time triggers surfaced as raw JSON.
- **Opt-in composition mandate** — the new MCP `instructions` string stays minimal by default. Opt in to the full Composing / Never-composed / Recommendation composition protocol with `VESTIGE_SYSTEM_PROMPT_MODE=full` when you want it, and nothing is imposed on your sessions when you don't.

## What's New in v2.0.5 "Intentional Amnesia"

**The first shipped AI memory system with top-down inhibitory control over retrieval.** Other systems implement passive decay — memories fade if you don't touch them. Vestige v2.0.5 also implements *active* suppression: the new **`suppress`** tool compounds a retrieval penalty on every call (up to 80%), a background Rac1 worker fades co-activated neighbors over 72 hours, and the whole thing is reversible within a 24-hour labile window. **Never deletes.** The memory is inhibited, not erased.

Ebbinghaus 1885 models what happens to memories you don't touch. Anderson 2025 models what happens when you actively want to stop thinking about one. Every other AI memory system implements the first. Vestige is the first to ship the second.

Based on [Anderson et al. 2025](https://www.nature.com/articles/s41583-025-00929-y) (Suppression-Induced Forgetting, *Nat Rev Neurosci*) and [Cervantes-Sandoval et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7477079/) (Rac1 synaptic cascade). **24 tools · 29 cognitive modules · 1,292 tests.**

<details>
<summary>Earlier releases (v2.0 "Cognitive Leap" → v2.0.4 "Deep Reference")</summary>

- **v2.0.4 — `deep_reference` Tool** — 8-stage cognitive reasoning pipeline with FSRS-6 trust scoring, intent classification, spreading activation, contradiction analysis, and pre-built reasoning chains. Token budgets raised 10K → 100K. CORS tightened.
- **v2.0 — 3D Memory Dashboard** — SvelteKit + Three.js neural visualization with real-time WebSocket events, bloom post-processing, force-directed graph layout.
- **v2.0 — WebSocket Event Bus** — Every cognitive operation broadcasts events: memory creation, search, dreaming, consolidation, retention decay.
- **v2.0 — HyDE Query Expansion** — Template-based Hypothetical Document Embeddings for dramatically improved search quality on conceptual queries.
- **v2.0 — Nomic v2 MoE (experimental)** — fastembed 5.11 with optional Nomic Embed Text v2 MoE (475M params, 8 experts) + Metal GPU acceleration.
- **v2.0 — Command Palette** — `Cmd+K` navigation, keyboard shortcuts, responsive mobile layout, PWA installable.
- **v2.0 — FSRS Decay Visualization** — SVG retention curves with predicted decay at 1d/7d/30d.

</details>

---

## Quick Start

```bash
# 1. Install (macOS Apple Silicon)
curl -L https://github.com/samvallad33/vestige/releases/latest/download/vestige-mcp-aarch64-apple-darwin.tar.gz | tar -xz
sudo mv vestige-mcp vestige vestige-restore /usr/local/bin/

# 2. Connect to Claude Code
claude mcp add vestige vestige-mcp -s user

# Or connect to Codex
codex mcp add vestige -- /usr/local/bin/vestige-mcp

# 3. Test it
# "Remember that I prefer TypeScript over JavaScript"
# ...new session...
# "What are my coding preferences?"
# → "You prefer TypeScript over JavaScript."
```

<details>
<summary>Other platforms & install methods</summary>

**Linux (x86_64):**
```bash
curl -L https://github.com/samvallad33/vestige/releases/latest/download/vestige-mcp-x86_64-unknown-linux-gnu.tar.gz | tar -xz
sudo mv vestige-mcp vestige vestige-restore /usr/local/bin/
```

**macOS (Intel):** Microsoft is discontinuing x86_64 macOS prebuilts after ONNX Runtime v1.23.0, so Vestige's Intel Mac build links dynamically against a Homebrew-installed ONNX Runtime via the `ort-dynamic` feature. Install with:

```bash
brew install onnxruntime
curl -L https://github.com/samvallad33/vestige/releases/latest/download/vestige-mcp-x86_64-apple-darwin.tar.gz | tar -xz
sudo mv vestige-mcp vestige vestige-restore /usr/local/bin/
echo 'export ORT_DYLIB_PATH="'"$(brew --prefix onnxruntime)"'/lib/libonnxruntime.dylib"' >> ~/.zshrc
source ~/.zshrc
claude mcp add vestige vestige-mcp -s user
```

Full Intel Mac guide (build-from-source + troubleshooting): [`docs/INSTALL-INTEL-MAC.md`](docs/INSTALL-INTEL-MAC.md).

**Windows:** Prebuilt binaries ship but `usearch 2.24.0` hit an MSVC compile break ([usearch#746](https://github.com/unum-cloud/usearch/issues/746)); we've pinned `=2.23.0` until upstream fixes it. Source builds work with:

```bash
git clone https://github.com/samvallad33/vestige && cd vestige
cargo build --release -p vestige-mcp
```

**npm:**
```bash
npm install -g vestige-mcp-server
```

**Build from source (requires Rust 1.91+):**
```bash
git clone https://github.com/samvallad33/vestige && cd vestige
cargo build --release -p vestige-mcp
# Optional: enable Metal GPU acceleration on Apple Silicon
cargo build --release -p vestige-mcp --features metal
```
</details>

---

## Works Everywhere

Vestige speaks MCP — the universal protocol for AI tools. One brain, every IDE.

| IDE | Setup |
|-----|-------|
| **Claude Code** | `claude mcp add vestige vestige-mcp -s user` |
| **Codex** | [Integration guide](docs/integrations/codex.md) |
| **Claude Desktop** | [2-min setup](docs/CONFIGURATION.md#claude-desktop-macos) |
| **Xcode 26.3** | [Integration guide](docs/integrations/xcode.md) |
| **Cursor** | [Integration guide](docs/integrations/cursor.md) |
| **VS Code (Copilot)** | [Integration guide](docs/integrations/vscode.md) |
| **JetBrains** | [Integration guide](docs/integrations/jetbrains.md) |
| **Windsurf** | [Integration guide](docs/integrations/windsurf.md) |

---

## 🧠 3D Memory Dashboard

Vestige v2.0 ships with a real-time 3D visualization of your AI's memory. Every memory is a glowing node in 3D space. Watch connections form, memories pulse when accessed, and the entire graph come alive during dream consolidation.

**Features:**
- Force-directed 3D graph with 1000+ nodes at 60fps
- Bloom post-processing for cinematic neural network aesthetic
- Real-time WebSocket events: memories pulse on access, burst on creation, fade on decay
- Dream visualization: graph enters purple dream mode, replayed memories light up sequentially
- FSRS retention curves: see predicted memory decay at 1d, 7d, 30d
- Command palette (`Cmd+K`), keyboard shortcuts, responsive mobile layout
- Installable as PWA for quick access

**Tech:** SvelteKit 2 + Svelte 5 + Three.js + Tailwind CSS 4 + WebSocket

The dashboard runs automatically at `http://localhost:3927/dashboard` when the MCP server starts.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  SvelteKit Dashboard (apps/dashboard)                │
│  Three.js 3D Graph · WebGL + Bloom · Real-time WS   │
├─────────────────────────────────────────────────────┤
│  Axum HTTP + WebSocket Server (port 3927)            │
│  15 REST endpoints · WS event broadcast              │
├─────────────────────────────────────────────────────┤
│  MCP Server (stdio JSON-RPC)                         │
│  24 tools · 29 cognitive modules                     │
├─────────────────────────────────────────────────────┤
│  Cognitive Engine                                    │
│  ┌──────────┐ ┌──────────┐ ┌───────────────┐       │
│  │ FSRS-6   │ │ Spreading│ │ Prediction    │       │
│  │ Scheduler│ │ Activation│ │ Error Gating  │       │
│  └──────────┘ └──────────┘ └───────────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌───────────────┐       │
│  │ Memory   │ │ Synaptic │ │ Hippocampal   │       │
│  │ Dreamer  │ │ Tagging  │ │ Index         │       │
│  └──────────┘ └──────────┘ └───────────────┘       │
├─────────────────────────────────────────────────────┤
│  Storage Layer                                       │
│  SQLite + FTS5 · USearch HNSW · Nomic Embed v1.5    │
│  Optional: Nomic v2 MoE · Qwen3 Reranker · Metal   │
└─────────────────────────────────────────────────────┘
```

---

## Why Not Just Use RAG?

RAG is a dumb bucket. Vestige is an active organ.

| | RAG / Vector Store | Vestige |
|---|---|---|
| **Storage** | Store everything | **Prediction Error Gating** — only stores what's surprising or new |
| **Retrieval** | Nearest-neighbor | **7-stage pipeline** — HyDE expansion + reranking + spreading activation |
| **Decay** | Nothing expires | **FSRS-6** — memories fade naturally, context stays lean |
| **Forgetting** *(v2.0.5)* | Delete only | **`suppress` tool** — compounding top-down inhibition, neighbor cascade, reversible 24h |
| **Duplicates** | Manual dedup | **Self-healing** — auto-merges "likes dark mode" + "prefers dark themes" |
| **Importance** | All equal | **4-channel scoring** — novelty, arousal, reward, attention |
| **Sleep** | No consolidation | **Memory dreaming** — replays, connects, synthesizes insights |
| **Health** | No visibility | **Retention dashboard** — distributions, trends, recommendations |
| **Visualization** | None | **3D neural graph** — real-time WebSocket-powered Three.js |
| **Privacy** | Usually cloud | **100% local** — your data never leaves your machine |

---

## 🔬 The Cognitive Science Stack

This isn't a key-value store with an embedding model bolted on. Vestige implements real neuroscience:

**Prediction Error Gating** — The hippocampal bouncer. When new information arrives, Vestige compares it against existing memories. Redundant? Merged. Contradictory? Superseded. Novel? Stored with high synaptic tag priority.

**FSRS-6 Spaced Repetition** — 21 parameters governing the mathematics of forgetting. Frequently-used memories stay strong. Unused memories naturally decay. Your context window stays clean.

**HyDE Query Expansion** *(v2.0)* — Template-based Hypothetical Document Embeddings. Expands queries into 3-5 semantic variants, embeds all variants, and searches with the centroid embedding for dramatically better recall on conceptual queries.

**Synaptic Tagging** — A memory that seemed trivial this morning can be retroactively tagged as critical tonight. Based on [Frey & Morris, 1997](https://doi.org/10.1038/385533a0).

**Spreading Activation** — Search for "auth bug" and find the related JWT library update from last week. Memories form a graph, not a flat list. Based on [Collins & Loftus, 1975](https://doi.org/10.1037/0033-295X.82.6.407).

**Dual-Strength Model** — Every memory has storage strength (encoding quality) and retrieval strength (accessibility). A deeply stored memory can be temporarily hard to retrieve — just like real forgetting. Based on [Bjork & Bjork, 1992](https://doi.org/10.1016/S0079-7421(08)60016-9).

**Memory Dreaming** — Like sleep consolidation. Replays recent memories to discover hidden connections, strengthen important patterns, and synthesize insights. Dream-discovered connections persist to a graph database. Based on the [Active Dreaming Memory](https://engrxiv.org/preprint/download/5919/9826/8234) framework.

**Waking SWR Tagging** — Promoted memories get sharp-wave ripple tags for preferential replay during dream consolidation. 70/30 tagged-to-random ratio. Based on [Buzsaki, 2015](https://doi.org/10.1038/nn.3963).

**Autonomic Regulation** — Self-regulating memory health. Auto-promotes frequently accessed memories. Auto-GCs low-retention memories. Consolidation triggers on 6h staleness or 2h active use.

**Active Forgetting** *(v2.0.5)* — Top-down inhibitory control via the `suppress` tool. Other memory systems implement passive decay — the Ebbinghaus 1885 "use it or lose it" curve, sometimes with trust-weighted strength factors. Vestige v2.0.5 also implements *active* top-down suppression: each `suppress` call compounds (Suppression-Induced Forgetting, Anderson 2025), a background Rac1 cascade worker fades co-activated neighbors across the connection graph (Cervantes-Sandoval & Davis 2020), and a 24-hour labile window allows reversal (Nader reconsolidation semantics on a pragmatic axis). The memory persists — it's **inhibited, not erased**. Explicitly distinct from Anderson 1994 retrieval-induced forgetting (bottom-up, passive competition during retrieval), which is a separate, older primitive that several other memory systems implement. Based on [Anderson et al., 2025](https://www.nature.com/articles/s41583-025-00929-y) and [Cervantes-Sandoval et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7477079/). First shipped AI memory system with this primitive.

[Full science documentation ->](docs/SCIENCE.md)

---

## 🛠 24 MCP Tools

### Context Packets
| Tool | What It Does |
|------|-------------|
| `session_context` | **One-call session init** — replaces 5 calls with token-budgeted context, automation triggers, expandable IDs |

### Core Memory
| Tool | What It Does |
|------|-------------|
| `search` | 7-stage cognitive search — HyDE expansion + keyword + semantic + reranking + temporal + competition + spreading activation |
| `smart_ingest` | Intelligent storage with CREATE/UPDATE/SUPERSEDE via Prediction Error Gating. Batch mode for session-end saves |
| `memory` | Get, delete, check state, promote (thumbs up), demote (thumbs down) |
| `codebase` | Remember code patterns and architectural decisions per-project |
| `intention` | Prospective memory — "remind me to X when Y happens" |

### Cognitive Engine
| Tool | What It Does |
|------|-------------|
| `dream` | Memory consolidation — replays memories, discovers connections, synthesizes insights, persists graph |
| `explore_connections` | Graph traversal — reasoning chains, associations, bridges between memories |
| `predict` | Proactive retrieval — predicts what you'll need next based on context and activity |

### Autonomic
| Tool | What It Does |
|------|-------------|
| `memory_health` | Retention dashboard — distribution, trends, recommendations |
| `memory_graph` | Knowledge graph export — force-directed layout, up to 200 nodes |

### Scoring & Dedup
| Tool | What It Does |
|------|-------------|
| `importance_score` | 4-channel neuroscience scoring (novelty, arousal, reward, attention) |
| `find_duplicates` | Detect and merge redundant memories via cosine similarity |

### Maintenance
| Tool | What It Does |
|------|-------------|
| `system_status` | Combined health + stats + cognitive state + recommendations |
| `consolidate` | Run FSRS-6 decay cycle (also auto-runs every 6 hours) |
| `memory_timeline` | Browse chronologically, grouped by day |
| `memory_changelog` | Audit trail of state transitions |
| `backup` / `export` / `gc` | Database backup, JSON export, garbage collection |
| `restore` | Restore from JSON backup |

### Deep Reference (v2.0.4)
| Tool | What It Does |
|------|-------------|
| `deep_reference` | **Cognitive reasoning across memories.** 8-stage pipeline: FSRS-6 trust scoring, intent classification, spreading activation, temporal supersession, contradiction analysis, relation assessment, dream insight integration, and algorithmic reasoning chain generation. Returns trust-scored evidence with a pre-built reasoning scaffold. |
| `cross_reference` | Backward-compatible alias for `deep_reference`. |

### Active Forgetting (v2.0.5)
| Tool | What It Does |
|------|-------------|
| `suppress` | **Top-down active forgetting** — neuroscience-grounded inhibitory control over retrieval. Distinct from `memory.delete` (destroys the row) and `memory.demote` (one-shot ranking hit). Each call **compounds** a retrieval-score penalty (Anderson 2025 SIF), and a background Rac1 cascade worker fades co-activated neighbors over 72h (Davis 2020). Reversible within a 24-hour labile window via `reverse: true`. **The memory persists** — it is inhibited, not erased. |

---

## Make Your AI Use Vestige Automatically

Add this to your `CLAUDE.md`:

```markdown
## Memory

At the start of every session:
1. Search Vestige for user preferences and project context
2. Save bug fixes, decisions, and patterns without being asked
3. Create reminders when the user mentions deadlines
```

| You Say | AI Does |
|---------|---------|
| "Remember this" | Saves immediately |
| "I prefer..." / "I always..." | Saves as preference |
| "Remind me..." | Creates a future trigger |
| "This is important" | Saves + promotes |

[Full CLAUDE.md templates ->](docs/CLAUDE-SETUP.md)

---

## Technical Details

| Metric | Value |
|--------|-------|
| **Language** | Rust 2024 edition (MSRV 1.91) |
| **Codebase** | 80,000+ lines, 1,292 tests (366 core + 425 mcp + 497 e2e + 4 doctests) |
| **Binary size** | ~20MB |
| **Embeddings** | Nomic Embed Text v1.5 (768d → 256d Matryoshka, 8192 context) |
| **Vector search** | USearch HNSW (20x faster than FAISS) |
| **Reranker** | Jina Reranker v1 Turbo (38M params, +15-20% precision) |
| **Storage** | SQLite + FTS5 (optional SQLCipher encryption) |
| **Dashboard** | SvelteKit 2 + Svelte 5 + Three.js + Tailwind CSS 4 |
| **Transport** | MCP stdio (JSON-RPC 2.0) + WebSocket |
| **Cognitive modules** | 30 stateful (17 neuroscience, 11 advanced, 2 search) |
| **First run** | Downloads embedding model (~130MB), then fully offline |
| **Platforms** | macOS ARM + Intel + Linux x86_64 + Windows x86_64 (all prebuilt). Intel Mac needs `brew install onnxruntime` — see [install guide](docs/INSTALL-INTEL-MAC.md). |

### Optional Features

```bash
# Metal GPU acceleration (Apple Silicon — faster embedding inference)
cargo build --release -p vestige-mcp --features metal

# Nomic Embed Text v2 MoE (475M params, 305M active, 8 experts)
cargo build --release -p vestige-mcp --features nomic-v2

# Qwen3 Reranker (Candle backend, high-precision cross-encoder)
cargo build --release -p vestige-mcp --features qwen3-reranker

# SQLCipher encryption
cargo build --release -p vestige-mcp --no-default-features --features encryption,embeddings,vector-search
```

---

## CLI

```bash
vestige stats                    # Memory statistics
vestige stats --tagging          # Retention distribution
vestige stats --states           # Cognitive state breakdown
vestige health                   # System health check
vestige consolidate              # Run memory maintenance
vestige restore <file>           # Restore from backup
vestige dashboard                # Open 3D dashboard in browser
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [FAQ](docs/FAQ.md) | 30+ common questions answered |
| [Science](docs/SCIENCE.md) | The neuroscience behind every feature |
| [Storage Modes](docs/STORAGE.md) | Global, per-project, multi-instance |
| [CLAUDE.md Setup](docs/CLAUDE-SETUP.md) | Templates for proactive memory |
| [Configuration](docs/CONFIGURATION.md) | CLI commands, environment variables |
| [Integrations](docs/integrations/) | Codex, Xcode, Cursor, VS Code, JetBrains, Windsurf |
| [Changelog](CHANGELOG.md) | Version history |

---

## Troubleshooting

<details>
<summary>"Command not found" after installation</summary>

Ensure `vestige-mcp` is in your PATH:
```bash
which vestige-mcp
# Or use the full path:
claude mcp add vestige /usr/local/bin/vestige-mcp -s user
```
</details>

<details>
<summary>Embedding model download fails</summary>

First run downloads ~130MB from Hugging Face. If behind a proxy:
```bash
export HTTPS_PROXY=your-proxy:port
```

Cache: macOS `~/Library/Caches/com.vestige.core/fastembed` | Linux `~/.cache/vestige/fastembed`
</details>

<details>
<summary>Dashboard not loading</summary>

The dashboard starts automatically on port 3927 when the MCP server runs. Check:
```bash
curl http://localhost:3927/api/health
# Should return {"status":"healthy",...}
```
</details>

[More troubleshooting ->](docs/FAQ.md#troubleshooting)

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

AGPL-3.0 — free to use, modify, and self-host. If you offer Vestige as a network service, you must open-source your modifications.

---

<p align="center">
  <i>Built by <a href="https://github.com/samvallad33">@samvallad33</a></i><br>
  <sub>80,000+ lines of Rust · 29 cognitive modules · 130 years of memory research · one 22MB binary</sub>
</p>
