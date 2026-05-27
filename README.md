<div align="center">

# Vestige

### Local cognitive memory for MCP-compatible AI agents.

[![GitHub stars](https://img.shields.io/github/stars/samvallad33/vestige?style=social)](https://github.com/samvallad33/vestige)
[![Release](https://img.shields.io/github/v/release/samvallad33/vestige)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/samvallad33/vestige/actions)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green)](https://modelcontextprotocol.io)

**Your agent forgets project decisions between sessions. Vestige gives it local, inspectable memory.**

Built on proven memory and retrieval ideas — FSRS-6 spaced repetition, prediction error gating, synaptic tagging, spreading activation, and memory consolidation — all running in a single Rust binary with a local dashboard. 100% local. Zero cloud.

[Quick Start](#quick-start) | [Dashboard](#-3d-memory-dashboard) | [How It Works](#-the-cognitive-science-stack) | [Tools](#-25-mcp-tools) | [Docs](docs/)

</div>

---

## What's New in v2.1.23 "Receipt Lock Hardening"

v2.1.23 turns the Sanhedrin Receipt Lock launch into something more portable,
observable, and harder to spoof.

- **Model-agnostic Sanhedrin presets.** Sanhedrin no longer guesses a large default verifier. Users choose any OpenAI-compatible endpoint/model, or start from custom, small laptop, Ollama, MLX, vLLM, llama.cpp, hosted API, or LiteLLM presets.
- **Sharper Receipt Lock.** Verification claims inside code fences, quotes, blockquotes, or explicitly hedged "let me verify" language no longer trigger false vetoes, while actual "tests passed" claims still require command receipts.
- **Safer command receipts.** Transcript command evidence now prefers structured tool-use receipts; loose JSON scanning is opt-in only.
- **Visible fail-open telemetry.** Timeouts, unavailable model endpoints, and malformed verdicts are logged locally and surfaced in the dashboard's 7-day Sanhedrin stats.
- **Durable evidence boundary.** Staged evidence remains useful context, but it cannot satisfy durable support or contradiction requirements by itself.
- **Safer batch writes.** `smart_ingest` batch mode now keeps caller-separated items separate by default and returns merge previews when an existing memory is mutated.
- **Opt-in NVIDIA acceleration path.** Qwen3 embedding builds expose CUDA/cuDNN feature flags for contributors and users with CUDA-capable hosts.

## What's New in v2.1.22 "Sanhedrin Receipts"

v2.1.22 makes the optional Sanhedrin hook accountable enough to trust in daily
agent work. Vetoes now leave local receipts, verification claims need real
command evidence, and users can appeal stale or over-strict blocks from the
dashboard.

- **Receipt Lock.** Claims like "tests passed", "build is green", or "lint is clean" are blocked unless the current transcript contains a matching successful command receipt.
- **Screenshotable veto receipts.** Sanhedrin writes `~/.vestige/sanhedrin/latest.json` and `latest.html` with Claim -> Verdict -> Precedent -> Fix -> Appeal.
- **Dashboard Verdict Bar.** The dashboard shows PASS, NOTE, CAUTION, VETO, or APPEALED globally, expands into the receipt, and records stale/wrong/too-strict appeals.
- **Claim ledger.** Claim-mode Sanhedrin output now maps every extracted claim into structured JSON instead of treating the whole draft as one blob.
- **Appeal training.** Appeals are saved to `appeals.jsonl` and suppress future vetoes for the same claim fingerprint.

## What's New in v2.1.21 "Agent-Neutral Hardening"

v2.1.21 tightens Vestige for normal use across MCP-compatible agents, without
making Claude Code companion tooling part of the default path.

- **Agent-neutral default.** Stdio MCP remains the default transport; optional HTTP MCP is explicit with `--http`, `--http-port`, or `VESTIGE_HTTP_ENABLED=1`.
- **Safer destructive actions.** `memory(action="delete")` now requires `confirm=true`, matching `purge`, and the legacy `delete_knowledge` shim forwards that confirmation instead of bypassing it.
- **Portable sync repair.** Merge imports preserve purge tombstones, avoid `INSERT OR REPLACE` cascades, rebuild the vector index from a clean state, and write portable archive temp files with private Unix permissions.
- **Release/package cleanup.** Release builds check the embedded dashboard before packaging, publish checksums, and the npm installer rejects targets that do not have release assets.
- **Any-agent memory protocol.** The setup docs now include a short agent-agnostic memory protocol for Claude Code, Codex, Cursor, VS Code, Xcode, JetBrains, Windsurf, and other MCP clients.

## What's New in v2.1.2 "Honest Memory"

v2.1.2 makes Vestige easier to trust in everyday work: literal lookups stay literal, purge really removes content, contradictions are inspectable, and updates no longer require a curl reinstall flow.

- **Concrete search mode.** Quoted strings, env vars, UUIDs, paths, and code identifiers now take a keyword/literal path that skips HyDE, semantic fusion, FSRS reweighting, competition, and spreading activation. Exact things like `OPENAI_API_KEY`, `mlx_lm.server`, and migration IDs land first.
- **Irreversible purge.** `memory(action="purge", confirm=true)` permanently removes memory content and embeddings, scrubs insight JSON references, detaches temporal-summary children, prunes graph edges, and keeps only a non-content deletion tombstone for sync/audit.
- **First-class contradiction inspection.** New `contradictions` MCP tool surfaces trust-weighted disagreements directly instead of hiding them inside `deep_reference`.
- **Simple update flow.** `vestige update` refreshes binaries. Claude Code Cognitive Sandwich companion files are opt-in with `vestige update --sandwich-companion` or `vestige sandwich install`.
- **Pro waitlist preview.** `/dashboard/waitlist` adds a local-first Solo Pro and Team Pro early-access surface. `VITE_WAITLIST_ENDPOINT` and `VITE_SUPPORT_BOT_ENDPOINT` are opt-in dashboard env vars, so no signup data is captured unless endpoints are configured.

## What's New in v2.1.1 "Portable Sync"

v2.1.1 focuses on the biggest post-launch ask: move memories between machines without losing cognitive state. It also adds opt-in Qwen3 embeddings for higher-recall local retrieval.

- **Exact portable archives.** `vestige portable-export` / `vestige portable-import` preserve IDs, FSRS state, graph edges, suppression state, audit rows, and embedding blobs for Vestige-to-Vestige device transfer.
- **Sync-safe merge storage.** `vestige portable-import --merge` and `vestige sync <archive>` merge non-empty databases, apply delete tombstones, keep newer local memories, rebuild FTS, and push through a pluggable portable-sync backend. v2.1.1 ships the file backend for Dropbox, iCloud, Syncthing, Git, and shared folders.
- **Qwen3 embeddings.** Build with `qwen3-embeddings`, set `VESTIGE_EMBEDDING_MODEL=qwen3-0.6b`, and run `vestige consolidate` to re-embed existing memories. `vestige health` reports mixed-model stores before search quality is affected.
- **Model-aware retrieval.** Vestige now avoids comparing Qwen and Nomic vectors in the same search/dedup path.

## What's New in v2.1.0 "Cognitive Sandwich Goes Local"

v2.1.0 adds an opt-in Claude Code hook harness around the existing Vestige MCP server. The MCP tool surface and database schema stay backward compatible, while preflight hooks can inject trusted memory context before Claude answers. The heavyweight Sanhedrin verifier is optional and can be enabled separately.

- **Optional Sanhedrin Executioner.** The post-response verifier is off by default. Users can enable it with an OpenAI-compatible endpoint on x86/Linux/Intel Mac, or add `--with-launchd` on Apple Silicon to run the local MLX Qwen backend.
- **One-command Cognitive Sandwich installer.** `vestige sandwich install` stages hook files and agents by default, removes old Vestige hook wiring, and leaves all Claude Code hook layers plus the 19 GB model path opt-in.
- **Pulse hook backed by `/api/changelog`.** Fresh dream and connection events can be injected into the next Claude Code prompt context without blocking the prompt.
- **`VESTIGE_DATA_DIR` support.** `--data-dir` now has an env-var fallback, tilde expansion, secure directory creation, and clear precedence docs.
- **NPM release wrapper fixed.** `vestige-mcp-server@2.1.0` now downloads binaries from the matching `v2.1.0` GitHub release tag instead of an old hardcoded release.

## What's New in v2.0.9 "Autopilot"

Autopilot flips Vestige from passive memory library to **self-managing cognitive surface**. Same 24 MCP tools, zero schema changes — but the moment you upgrade, 14 previously dormant cognitive primitives start firing on live events without any tool call from your client.

- **One supervised backend task subscribes to the 20-event WebSocket bus** and routes six event classes into the cognitive engine: `MemoryCreated` triggers synaptic-tagging PRP + predictive-access records, `SearchPerformed` warms the speculative-retrieval model, `MemoryPromoted` fires activation spread, `MemorySuppressed` emits the Rac1 cascade wave, high-importance `ImportanceScored` (>0.85) auto-promotes, and `Heartbeat` rate-limit-fires `find_duplicates` on large DBs. **The engine mutex is never held across `.await`, so MCP dispatch is never starved.**
- **Panic-resilient supervisors.** Both background tasks run inside an outer supervisor loop — if one handler panics on a bad memory, the supervisor respawns it in 5 s instead of losing every future event.
- **Fully backward compatible.** No new MCP tools. No schema migration. Existing v2.0.8 databases open without a single step. Opt out with `VESTIGE_AUTOPILOT_ENABLED=0` if you want the passive-library contract back.
- **3,091 LOC of orphan v1.0 tool code removed** — nine superseded modules (`checkpoint`, `codebase`, `consolidate`, `ingest`, `intentions`, `knowledge`, `recall`, plus helpers) verified zero non-test callers before deletion. Tool surface unchanged.

## What's New in v2.0.8 "Pulse"

v2.0.8 wires the dashboard through to the cognitive engine. Eight new surfaces expose the reasoning stack visually — every one was MCP-only before.

- **Reasoning Theater (`/reasoning`)** — `Cmd+K` Ask palette over the 8-stage `deep_reference` pipeline (hybrid retrieval → cross-encoder rerank → spreading activation → FSRS-6 trust → temporal supersession → contradiction analysis → relation assessment → template reasoning chain). Evidence cards, confidence meter, contradiction geodesic arcs, superseded-memory lineage, evolution timeline. **Zero LLM calls, 100% local.**
- **Pulse InsightToast** — real-time toasts for `DreamCompleted`, `ConsolidationCompleted`, `ConnectionDiscovered`, promote/demote/suppress/unsuppress, `Rac1CascadeSwept`. Rate-limited, auto-dismiss, click-to-dismiss.
- **Memory Birth Ritual (Terrarium)** — new memories materialize in the 3D graph on every `MemoryCreated`: elastic scale-in, quadratic Bezier flight path, glow sprite fade-in, Newton's Cradle docking recoil. 60-frame sequence, zero-alloc math.
- **7 more dashboard surfaces** — `/duplicates`, `/dreams`, `/schedule`, `/importance`, `/activation`, `/contradictions`, `/patterns`. Left nav expanded 8 → 16 with single-key shortcuts.
- **Intel Mac (`x86_64-apple-darwin`) support restored** via the `ort-dynamic` Cargo feature + Homebrew `onnxruntime`. Microsoft deprecated x86_64 macOS prebuilts; the dynamic-link path sidesteps that permanently. **Closes #41.**
- **Contradiction-detection false positives eliminated** — four thresholds tightened so adjacent-domain memories no longer flag as conflicts. On an FSRS-6 query this collapses false contradictions 12 → 0 without regressing legitimate test cases.

## What's New in v2.0.7 "Visible"

Hygiene release closing two UI gaps and finishing schema cleanup. No breaking changes, no user-data migrations.

- **`POST /api/memories/{id}/suppress` + `/unsuppress` HTTP endpoints** — dashboard can trigger Anderson 2025 SIF + Rac1 cascade without dropping to raw MCP. `suppressionCount`, `retrievalPenalty`, `reversibleUntil`, `labileWindowHours` all in response. Suppress button joins Promote / Demote / Delete on the Memories page.
- **Uptime in the sidebar footer** — the `Heartbeat` event has carried `uptime_secs` since v2.0.5 but was never rendered. Now shows as `up 3d 4h` / `up 18m` / `up 47s`.
- **`execute_export` panic fix** — unreachable match arm replaced with a clean "unsupported export format" error instead of unwinding through the MCP dispatcher.
- **`predict` surfaces `predict_degraded: true`** on lock poisoning instead of silently returning empty vecs. `memory_changelog` honors `start` / `end` bounds. `intention` check honors `include_snoozed`.
- **Migration V11** — drops dead `knowledge_edges` + `compressed_memories` tables (added speculatively in V4, never used).

## What's New in v2.0.6 "Composer"

v2.0.6 is a polish release that makes the existing cognitive stack finally *feel* alive in the dashboard and stays out of your way on the prompt side.

- **Six live graph reactions, not one** — `MemorySuppressed`, `MemoryUnsuppressed`, `Rac1CascadeSwept`, `Connected`, `ConsolidationStarted`, and `ImportanceScored` now light the 3D graph in real time. v2.0.5 shipped `suppress` but the graph was silent when you called it; consolidation and importance scoring have been silent since v2.0.0. No longer.
- **Intentions page actually works** — fixes a long-standing bug where every intention rendered as "normal priority" (type/schema drift between backend and frontend) and context/time triggers surfaced as raw JSON.
- **Opt-in composition mandate** — the new MCP `instructions` string stays minimal by default. Opt in to the full Composing / Never-composed / Recommendation composition protocol with `VESTIGE_SYSTEM_PROMPT_MODE=full` when you want it, and nothing is imposed on your sessions when you don't.

## What's New in v2.0.5 "Intentional Amnesia"

**The first shipped AI memory system with top-down inhibitory control over retrieval.** Other systems implement passive decay — memories fade if you don't touch them. Vestige v2.0.5 also implements *active* suppression: the new **`suppress`** tool compounds a retrieval penalty on every call (up to 80%), a background Rac1 worker fades co-activated neighbors over 72 hours, and the whole thing is reversible within a 24-hour labile window. **Never deletes.** The memory is inhibited, not erased.

Ebbinghaus 1885 models what happens to memories you don't touch. Anderson 2025 models what happens when you actively want to stop thinking about one. Every other AI memory system implements the first. Vestige is the first to ship the second.

Based on [Anderson et al. 2025](https://www.nature.com/articles/s41583-025-00929-y) (Suppression-Induced Forgetting, *Nat Rev Neurosci*) and [Cervantes-Sandoval et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7477079/) (Rac1 synaptic cascade).

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
# 1. Install
npm install -g vestige-mcp-server@latest

# 2. Connect to any MCP-compatible agent
# Claude Code
claude mcp add vestige vestige-mcp -s user

# Codex
codex mcp add vestige -- vestige-mcp

# 3. Test it
# "Remember that I prefer TypeScript over JavaScript"
# ...new session...
# "What are my coding preferences?"
# → "You prefer TypeScript over JavaScript."
```

<details>
<summary>Other platforms & install methods</summary>

**Updating an existing install:**
```bash
vestige update
```

`vestige update` updates only the Vestige binaries by default. Use
`vestige update --sandwich-companion` if you also want to refresh optional Claude
Code Cognitive Sandwich companion files.

**macOS/Linux manual binary install:**
```bash
vestige update --install-dir /usr/local/bin
```

**macOS (Intel):** Microsoft is discontinuing x86_64 macOS prebuilts after ONNX Runtime v1.23.0, so Vestige's Intel Mac build links dynamically against a Homebrew-installed ONNX Runtime via the `ort-dynamic` feature. Install with:

```bash
brew install onnxruntime
npm install -g vestige-mcp-server@latest
echo 'export ORT_DYLIB_PATH="'"$(brew --prefix onnxruntime)"'/lib/libonnxruntime.dylib"' >> ~/.zshrc
source ~/.zshrc
claude mcp add vestige vestige-mcp -s user
```

Full Intel Mac guide (build-from-source + troubleshooting): [`docs/INSTALL-INTEL-MAC.md`](docs/INSTALL-INTEL-MAC.md).

**Windows + Claude Desktop (recommended):**

Fully quit Claude Desktop from the system tray, then install or update Vestige from PowerShell:

```powershell
npm install -g vestige-mcp-server@latest
vestige-mcp --version
```

Open `%APPDATA%\Claude\claude_desktop_config.json` and point Claude Desktop at the installed MCP command:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

If Claude Desktop cannot find `vestige-mcp`, run `where vestige-mcp` in PowerShell and use the exact `.cmd` path it prints as `command`. Example: `"C:\\Users\\you\\AppData\\Roaming\\npm\\vestige-mcp.cmd"`. Reopen Claude Desktop after saving. Future binary updates use `vestige update`; optional Claude Code companion files require `vestige update --sandwich-companion`.

**Windows source build:** Prebuilt binaries ship but `usearch 2.24.0` hit an MSVC compile break ([usearch#746](https://github.com/unum-cloud/usearch/issues/746)); we've pinned `=2.23.0` until upstream fixes it. Source builds work with:

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

Vestige speaks MCP, so any client that can register a stdio MCP server can use it.

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

Run `vestige dashboard` to open `http://localhost:3927/dashboard`, or set `VESTIGE_DASHBOARD_ENABLED=true` to start it with the MCP server.

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
│  25 tools · 30 cognitive modules                     │
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

## 🛠 25 MCP Tools

### Context Packets
| Tool | What It Does |
|------|-------------|
| `session_context` | **One-call session init** — replaces 5 calls with token-budgeted context, automation triggers, expandable IDs |

### Core Memory
| Tool | What It Does |
|------|-------------|
| `search` | Concrete literal search for exact identifiers, or 7-stage cognitive search — HyDE expansion + keyword + semantic + reranking + temporal + competition + spreading activation |
| `smart_ingest` | Intelligent storage with CREATE/UPDATE/SUPERSEDE via Prediction Error Gating. Batch mode for session-end saves |
| `memory` | Get, purge content/embeddings, check state, promote (thumbs up), demote (thumbs down), edit |
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
| `backup` / `export` / `gc` | Database backup, JSON/JSONL/portable export, garbage collection |
| `restore` | Restore from JSON backup or portable archive |

### Deep Reference (v2.0.4)
| Tool | What It Does |
|------|-------------|
| `deep_reference` | **Cognitive reasoning across memories.** 8-stage pipeline: FSRS-6 trust scoring, intent classification, spreading activation, temporal supersession, contradiction analysis, relation assessment, dream insight integration, and algorithmic reasoning chain generation. Returns trust-scored evidence with a pre-built reasoning scaffold. |
| `cross_reference` | Backward-compatible alias for `deep_reference`. |
| `contradictions` | **Honest memory inspection.** Scans a topic or recent memories for trust-weighted disagreements using the same local contradiction logic as `deep_reference`. |

### Active Forgetting (v2.0.5)
| Tool | What It Does |
|------|-------------|
| `suppress` | **Top-down active forgetting** — neuroscience-grounded inhibitory control over retrieval. Distinct from `memory(action="purge")`, which permanently removes content/embeddings. Each suppression compounds a retrieval-score penalty (Anderson 2025 SIF), and a background Rac1 cascade worker fades co-activated neighbors over 72h (Davis 2020). Reversible within a 24-hour labile window via `reverse: true`. **The memory persists** — it is inhibited, not erased. |

---

## Make Your AI Use Vestige Automatically

Registering the MCP server exposes tools; the agent still needs an instruction
that tells it when to call memory. Use the agent-neutral protocol, then adapt it
to your client-specific instruction file.

| You Say | AI Does |
|---------|---------|
| "Remember this" | Saves immediately |
| "I prefer..." / "I always..." | Saves as preference |
| "Remind me..." | Creates a future trigger |
| "This is important" | Saves + promotes |

[Agent memory protocol ->](docs/AGENT-MEMORY-PROTOCOL.md) · [Claude Code template ->](docs/CLAUDE-SETUP.md)

---

## Technical Details

| Metric | Value |
|--------|-------|
| **Language** | Rust 2024 edition (MSRV 1.91) |
| **Codebase** | 80,000+ lines with Rust core/MCP/e2e, dashboard, and hook coverage |
| **Binary size** | ~20MB |
| **Embeddings** | Nomic Embed Text v1.5 by default (768d -> 256d Matryoshka, 8192 context); Qwen3 0.6B optional |
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
# Qwen3 embeddings (Candle backend; add metal on Apple Silicon)
cargo build --release -p vestige-mcp --features qwen3-embeddings,metal
VESTIGE_EMBEDDING_MODEL=qwen3-0.6b vestige consolidate
```

### Building with CUDA support (NVIDIA hosts - Windows / Linux)

The `cuda` feature routes Qwen3 embedding through NVIDIA GPUs via
`candle-core/cuda`. On a host with the CUDA toolkit installed and a supported
NVIDIA runtime, this drops Qwen3-Embedding inference from CPU-bound to GPU-bound
for batched workloads.

```bash
# Linux / Windows + CUDA toolkit (12.x or 13.x)
cargo build --release -p vestige-mcp --features qwen3-embeddings,cuda

# Optional cuDNN acceleration on top of CUDA
cargo build --release -p vestige-mcp --features qwen3-embeddings,cudnn

VESTIGE_EMBEDDING_MODEL=qwen3-0.6b vestige consolidate
```

**Prerequisites:**

- NVIDIA driver + CUDA toolkit (12.x or 13.x). Verify with `nvcc --version`.
- A C++ host compiler that `nvcc` can drive (Linux: `gcc`; Windows: MSVC /
  `cl.exe` from a recent Visual Studio Build Tools install).

**Windows + MSVC + CUDA 13.x build note.** Recent CCCL headers shipped with
CUDA 13.x require the modern preprocessor. Without it, the `candle-kernels`
`.cu` compile pass can fail at `cuda/include/cuda/std/__cccl/compiler.h`. Set
this env var before `cargo build` to pass `/Zc:preprocessor` through `nvcc`:

```powershell
# PowerShell
$env:NVCC_PREPEND_FLAGS = '-Xcompiler="/Zc:preprocessor"'
cargo build --release -p vestige-mcp --features qwen3-embeddings,cuda
```

```cmd
:: cmd.exe
set NVCC_PREPEND_FLAGS=-Xcompiler="/Zc:preprocessor"
cargo build --release -p vestige-mcp --features qwen3-embeddings,cuda
```

Linux + CUDA 13.x builds with `gcc` do not need the equivalent flag.

**Verifying GPU is actually used.** With CUDA-enabled builds, run
`VESTIGE_EMBEDDING_MODEL=qwen3-0.6b vestige consolidate` on a corpus of 1000+
memories and watch `nvidia-smi`; embedding passes should pin a single GPU while
the run is active.

---

## CLI

```bash
vestige stats                    # Memory statistics
vestige stats --tagging          # Retention distribution
vestige stats --states           # Cognitive state breakdown
vestige health                   # System health check
vestige consolidate              # Run memory maintenance
vestige restore <file>           # Restore from backup
vestige portable-export <file>         # Exact cross-device archive
vestige portable-import <file>         # Import archive into an empty database
vestige portable-import <file> --merge # Merge archive into this database
vestige sync <file>                    # Pull/merge/push via file backend
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

Cache: platform user cache directory first, then `./.fastembed_cache` as a fallback. Override with `FASTEMBED_CACHE_PATH`.
</details>

<details>
<summary>Dashboard not loading</summary>

Run `vestige dashboard` or set `VESTIGE_DASHBOARD_ENABLED=true`, then check:
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
  <sub>80,000+ lines of Rust · 30 cognitive modules · 130 years of memory research · one 22MB binary</sub>
</p>
