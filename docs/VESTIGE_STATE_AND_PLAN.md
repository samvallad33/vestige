# Vestige: State of the Engine & Next-Phase Plan

> **For:** AI agents planning the next phase of Vestige alongside Sam.
> **From:** Sam Valladares (compiled via multi-agent inventory of the live codebase).
> **As of:** 2026-04-19 ~22:10 CT, post-merge of v2.0.8 into `main` (CI green on all four jobs).
> **Repo:** https://github.com/samvallad33/vestige
> **Related repo (private):** `~/Developer/vestige-cloud` (Feb 12, 2026 skeleton; Part 7).
>
> This document is the single authoritative briefing of what Vestige *is today* and what *ships next*. Everything in Part 1 is verifiable against the source tree; everything in Part 3 is the committed roadmap agreed 2026-04-19.

---

## Table of Contents

0. [Executive Summary (60-second read)](#0-executive-summary-60-second-read)
1. [What Vestige Is](#1-what-vestige-is)
2. [Workspace Architecture](#2-workspace-architecture)
3. [`vestige-core` — Cognitive Engine](#3-vestige-core--cognitive-engine)
4. [`vestige-mcp` — MCP Server + Dashboard Backend](#4-vestige-mcp--mcp-server--dashboard-backend)
5. [`apps/dashboard` — SvelteKit + Three.js Frontend](#5-appsdashboard--sveltekit--threejs-frontend)
6. [Integrations & Packaging](#6-integrations--packaging)
7. [`vestige-cloud` — Current Skeleton](#7-vestige-cloud--current-skeleton)
8. [Version History (v1.0 → v2.0.8)](#8-version-history-v10--v208)
9. [The Next-Phase Plan](#9-the-next-phase-plan)
10. [Composition Map](#10-composition-map)
11. [Risks & Known Gaps](#11-risks--known-gaps)
12. [Viral / Launch / Content Plan](#12-viral--launch--content-plan)
13. [How AI Agents Should Consume This Doc](#13-how-ai-agents-should-consume-this-doc)
14. [Glossary & Citations](#14-glossary--citations)

---

## 0. Executive Summary (60-second read)

Vestige is a Rust-based MCP (Model Context Protocol) cognitive memory server that gives any AI agent persistent, structured, scientifically-grounded memory. It ships three binaries (`vestige-mcp`, `vestige`, `vestige-restore`), a 3D SvelteKit dashboard embedded into the binary, and is distributed via GitHub releases + npm. As of v2.0.7 "Visible" (tagged 2026-04-19), it has **24 MCP tools**, **29 cognitive modules** implementing real neuroscience (FSRS-6 spaced repetition, synaptic tagging, hippocampal indexing, spreading activation, reconsolidation, Anderson 2025 suppression-induced forgetting, Rac1 cascade decay), **1,292 Rust tests**, **251 dashboard tests** (80 just added for v2.0.8 colour-mode), and **402 GitHub stars**. AGPL-3.0.

**The branch `feat/v2.0.8-memory-state-colors` was fast-forwarded into `main` tonight** adding the FSRS memory-state colour mode, a floating legend, ruthless unit coverage, the Rust 1.95 clippy-compat fix (12 sites), and the dark-glass-pill label redesign. CI on main: all 4 jobs ✅.

**The next six releases are scoped:** v2.1 "Decide" (Qwen3 embeddings, in-flight on `feat/v2.1.0-qwen3-embed`), v2.2 "Pulse" (subconscious cross-pollination — **the viral moment**), v2.3 "Rewind" (temporal slider + pin), v2.4 "Empathy" (emotional/frustration tagging, **first Pro-tier gate candidate**), v2.5 "Grip" (neuro-feedback cluster gestures), v2.6 "Remote" (`vestige-cloud` upgrade from 5→24 MCP tools + Streamable HTTP). v3.0 "Branch" reserves CoW memory branching and multi-tenant SaaS.

**Sam's context** (load-bearing for any strategic advice): no steady income since March 2026, Mays Business School deadline May 1 ($400K+ prizes), Orbit Wars Kaggle deadline June 23 ($5K × top 10), graduation June 13. Viral OSS growth comes first; paid tier gates second.

---

## 1. What Vestige Is

### 1.1 Mission

Give any AI agent that speaks MCP a long-term memory and a reasoning co-processor that survives session boundaries, with retrieval ranked by scientifically-validated decay and strengthening rules — not a vector database with a nice coat of paint.

### 1.2 Positioning vs. the competitive landscape

| System | Vestige's angle |
|---|---|
| Zep, Cognee, Letta, claude-mem, MemPalace, HippoRAG | Vestige is **local-first + MCP-native + neuroscience-grounded**. The others are cloud-first (Zep/Cognee), RAG-wrappers (HippoRAG), or toy (claude-mem). Vestige is the only one that implements 29 stateful cognitive modules. |
| ChatGPT memory, Cursor memory | Both are opaque key-value caches owned by their vendor. Vestige is open source and the memory is yours. |
| Plain vector DBs (Chroma, Qdrant) | They retrieve by similarity. Vestige *rewires* the graph on access (testing effect), decays with FSRS-6, competes retrievals, and dreams between sessions. |

### 1.3 The "Oh My God" surface

1. The 3D graph that **animates in real-time** when memories are created, promoted, suppressed, or cascade-decayed.
2. The `dream()` tool that runs a 5-stage consolidation cycle and generates insights from cross-cluster replay.
3. `deep_reference` — an 8-stage cognitive reasoning pipeline with FSRS trust scoring, intent classification, contradiction analysis, and a pre-built reasoning chain. Not just retrieval — actual reasoning.
4. Active forgetting (v2.0.5 "Intentional Amnesia") — top-down inhibitory control with Rac1 cascade that spreads over 72h, reversible within a 24h labile window.
5. Cross-IDE persistence. Fix a bug in VS Code, open the project in Xcode, the agent remembers.

### 1.4 Stats (as of 2026-04-19 post-merge)

| Metric | Value |
|---|---|
| GitHub stars | 402 |
| Total commits (main) | 139 |
| Rust source LOC | ~42,000 (vestige-core) + ~vestige-mcp |
| Rust tests passing | 1,292 (workspace, release profile) |
| Dashboard tests passing | 251 (Vitest, 7 files, 3,291 lines) |
| MCP tools | 24 |
| Cognitive modules | 29 (16 neuroscience + 11 advanced + 2 search) |
| FSRS-6 trained parameters | 21 |
| Embedding dim (default) | 768 (nomic-embed-text-v1.5), truncatable to 256 (Matryoshka) |
| Binary targets shipped | 3 (aarch64-darwin, x86_64-linux, x86_64-windows) |
| IDE integrations documented | 8 (Claude Code, Claude Desktop, Cursor, VS Code Copilot, Codex, Xcode, JetBrains/Junie, Windsurf) |
| Latest GitHub release | v2.0.7 "Visible" (binaries up, npm pending Sam's Touch ID) |
| `main` HEAD | `30d92b5` (2026-04-19 21:52 CT) |
| CI on HEAD | All 4 jobs ✅ (Test macos, Test ubuntu, Release aarch64-darwin, Release x86_64-linux) |

### 1.5 License

**AGPL-3.0-only** (copyleft). If you run a modified Vestige as a network service, you must open-source your modifications. This is intentional — it protects against extract-and-host competitors while allowing a future commercial-license path for SaaS (Part 9.7).

---

## 2. Workspace Architecture

### 2.1 Repo layout

```
vestige/
├── Cargo.toml                       # Workspace root
├── Cargo.lock
├── pnpm-workspace.yaml              # pnpm monorepo marker
├── package.json                     # Root (v2.0.1, private)
├── .mcp.json                        # Self-registering MCP config
├── README.md                        # 22.5 KB marketing + quick-start
├── CHANGELOG.md                     # 31 KB, v1.0 → v2.0.7 Keep-a-Changelog format
├── CLAUDE.md                        # Project-level Claude instructions
├── CONTRIBUTING.md                  # Dev setup + test commands
├── SECURITY.md                      # Vuln reporting
├── LICENSE                          # AGPL-3.0 full text
├── crates/
│   ├── vestige-core/                # Library crate (cognitive engine)
│   └── vestige-mcp/                 # Binary crate (MCP server + dashboard backend)
├── apps/
│   └── dashboard/                   # SvelteKit 2 + Svelte 5 + Three.js frontend
├── packages/
│   ├── vestige-mcp-npm/             # npm: vestige-mcp-server (binary wrapper)
│   ├── vestige-init/                # npm: @vestige/init (zero-config installer)
│   └── vestige-mcpb/                # legacy, appears abandoned
├── tests/
│   └── vestige-e2e-tests/           # Integration tests over MCP protocol
├── docs/
│   ├── CLAUDE-SETUP.md
│   ├── CONFIGURATION.md
│   ├── FAQ.md
│   ├── SCIENCE.md
│   ├── STORAGE.md
│   ├── integrations/
│   │   ├── codex.md
│   │   ├── cursor.md
│   │   ├── jetbrains.md
│   │   ├── vscode.md
│   │   ├── windsurf.md
│   │   └── xcode.md
│   ├── launch/
│   │   ├── UI_ROADMAP_v2.1_v2.2.md  # compiled 2026-04-19
│   │   ├── show-hn.md
│   │   ├── blog-post.md
│   │   ├── demo-script.md
│   │   └── reddit-cross-reference.md
│   └── blog/
│       └── xcode-memory.md
├── scripts/
│   └── xcode-setup.sh               # 4.9 KB interactive installer
└── .github/
    └── workflows/
        ├── ci.yml                   # push-main + PR: clippy + test
        ├── release.yml              # tag push: binary build matrix
        └── test.yml                 # parallel unit/e2e/journey/dashboard/coverage
```

### 2.2 Dependency flow

```
┌─────────────────────┐
│  apps/dashboard     │  Svelte 5 + Three.js → static `build/`
│  (SvelteKit 2)      │  embedded via include_dir! into vestige-mcp binary
└──────────┬──────────┘
           │ HTTP / WebSocket
           ▼
┌─────────────────────┐       ┌──────────────────────┐
│  vestige-mcp        │ ────► │  vestige-core        │
│  (binary + dash BE) │       │  (cognitive engine)  │
│  Axum + JSON-RPC    │       │  FSRS-6, search,     │
│  MCP stdio + HTTP   │       │  embeddings, 29      │
│                     │       │  cognitive modules   │
└─────────────────────┘       └──────────────────────┘
                                       ▲
                                       │ path dep
                              ┌────────┴──────────┐
                              │  vestige-cloud    │ (separate repo, Feb 12
                              │  vestige-http     │  skeleton, not yet
                              │  (Axum + SSE)     │  shipped)
                              └───────────────────┘
```

### 2.3 Build profile

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
strip = true
opt-level = "z"        # Size-optimized; binary is ~22 MB with dashboard
```

### 2.4 Workspace Cargo.toml pinned version

Workspace `version = "2.0.5"`. Crate-level `Cargo.toml` files pin `2.0.7`. Version files are pumped together on each release (5 files: `crates/vestige-core/Cargo.toml`, `crates/vestige-mcp/Cargo.toml`, `apps/dashboard/package.json`, `packages/vestige-init/package.json`, `packages/vestige-mcp-npm/package.json`).

### 2.5 MSRV & editions

- **Rust MSRV:** 1.91 (enforced in `rust-version`).
- **CI Rust:** stable (currently 1.95 — which introduced the `unnecessary_sort_by` + `collapsible_match` lints tonight's fixes addressed).
- **Edition:** 2024 across the entire workspace.
- **Node:** 18+ for npm packages, 22+ for dashboard dev.
- **pnpm:** 10+ for workspace.

---

## 3. `vestige-core` — Cognitive Engine

### 3.1 Purpose

Library crate. Owns the entire cognitive engine: storage, FTS5, vector search, FSRS-6, embeddings, and the 29 cognitive modules. Has no knowledge of MCP, HTTP, or the dashboard — those live one crate up.

### 3.2 Metadata

```toml
name = "vestige-core"
version = "2.0.7"
edition = "2024"
rust-version = "1.91"
license = "AGPL-3.0-only"
description = "Cognitive memory engine - FSRS-6 spaced repetition, semantic embeddings, and temporal memory"
keywords = ["memory", "spaced-repetition", "fsrs", "embeddings", "knowledge-graph"]
```

### 3.3 Feature flags (8)

| Flag | Default | What it turns on | Cost |
|---|---|---|---|
| `embeddings` | **yes** | `mod embeddings`, fastembed v5.11, ONNX inference | +~130MB model download on first run |
| `vector-search` | **yes** | `mod search`, USearch HNSW, hybrid BM25 + semantic | negligible |
| `bundled-sqlite` | **yes** (mutex w/ `encryption`) | SQLite bundled via rusqlite 0.38 | +~2MB binary |
| `encryption` | no | SQLCipher encrypted DB | requires system libsqlcipher |
| `qwen3-reranker` | no | Qwen3 cross-encoder reranker | +candle-core deps |
| `qwen3-embed` | **no (v2.1 scaffolding)** | Qwen3 embed backend via Candle (Metal device + CPU fallback) | +candle-core, +~500MB Qwen3 model |
| `metal` | no | Metal GPU acceleration on Apple | macOS only |
| `nomic-v2` | no | Nomic Embed v2 MoE variant | +~200MB model |
| `ort-dynamic` | no | Runtime-load ORT instead of static prebuilt | required on glibc < 2.38 |

**Default feature set ships with embeddings + vector-search.** `qwen3-embed` is the v2.1 "Decide" scaffolding — dual-index with feature-gated `DEFAULT_DIMENSIONS` (1024 for Qwen3 vs 256 for Matryoshka-truncated Nomic).

### 3.4 Module tree (`src/lib.rs`)

```
src/
├── lib.rs                 # Module tree + prelude re-exports
├── prelude.rs             # KnowledgeNode, IngestInput, SearchResult, etc.
├── storage/               # SQLite + FTS5 + HNSW + migrations
│   ├── mod.rs             # Storage struct; public API
│   ├── sqlite.rs          # Connection setup, PRAGMAs, migrations
│   ├── migrations.rs      # V1..V11 migration chain (V11 dropped knowledge_edges + compressed_memories tables)
│   ├── schema.rs          # CREATE TABLE statements
│   ├── node.rs            # CRUD for KnowledgeNode
│   ├── edge.rs            # Edge insertion + deletion
│   ├── fts.rs             # FTS5 wrapper
│   ├── state_transitions.rs  # Append-only audit log
│   ├── consolidation_history.rs
│   ├── dream_history.rs
│   └── intention.rs       # Prospective memory persistence
├── search/                # 7-stage cognitive search pipeline
│   ├── mod.rs
│   ├── hybrid.rs          # BM25 + semantic fusion
│   ├── vector.rs          # USearch HNSW wrapper; DEFAULT_DIMENSIONS gated
│   ├── reranker.rs        # Jina Reranker v1 Turbo (38M params)
│   ├── temporal.rs        # Recency + validity window boosting
│   ├── context.rs         # Tulving 1973 encoding specificity
│   ├── competition.rs     # Anderson 1994 retrieval-induced forgetting
│   └── activation.rs      # Spreading activation side effects
├── embeddings/            # ONNX local + Qwen3 candle
│   ├── mod.rs             # EmbeddingService trait
│   ├── local.rs           # Backend enum (Nomic ONNX / Qwen3 Candle); metal device selection
│   ├── adaptive.rs        # AdaptiveEmbedder (Matryoshka 256/768/1024 tier)
│   ├── hyde.rs            # HyDE query expansion
│   └── cache.rs           # In-memory embedding LRU
├── fsrs/                  # Spaced repetition (21-param Anki FSRS-6)
│   ├── mod.rs
│   ├── params.rs          # Trained params
│   ├── algorithm.rs       # R(t) = (1 + factor × t / S)^(-w20)
│   └── review.rs          # apply_review
├── neuroscience/          # 16 modules (see §3.5)
│   ├── mod.rs
│   ├── activation.rs      # ActivationNetwork (Collins & Loftus 1975)
│   ├── synaptic_tagging.rs # SynapticTaggingSystem (Frey & Morris 1997)
│   ├── hippocampal_index.rs  # (Teyler & Rudy 2007)
│   ├── context_matcher.rs # (Tulving 1973)
│   ├── accessibility.rs   # AccessibilityCalculator
│   ├── competition.rs     # CompetitionManager (Anderson 1994)
│   ├── state_update.rs    # StateUpdateService
│   ├── importance_signals.rs # 4-channel (novelty/arousal/reward/attention)
│   ├── emotional_memory.rs # Brown & Kulik 1977 flashbulb memory
│   ├── predictive_retrieval.rs # Friston Free Energy 2010
│   ├── prospective_memory.rs # Intention fulfillment
│   ├── intention_parser.rs
│   └── memory_states.rs   # Active / Dormant / Silent / Unavailable + Bjork & Bjork 1992
├── advanced/              # 11 modules (see §3.6)
│   ├── mod.rs
│   ├── importance_tracker.rs
│   ├── reconsolidation.rs # Nader 2000 labile window (5 min, 10 mods max)
│   ├── intent_detector.rs # 9 intent types
│   ├── activity_tracker.rs
│   ├── dreaming.rs        # MemoryDreamer 5-stage
│   ├── chains.rs          # MemoryChainBuilder (A*-like)
│   ├── compression.rs     # MemoryCompressor (30-day min age)
│   ├── cross_project.rs   # CrossProjectLearner (6 pattern types)
│   ├── adaptive_embedding.rs
│   ├── speculative_retriever.rs
│   └── consolidation_scheduler.rs
├── codebase/              # CrossProjectLearner backing
│   ├── git.rs             # Git history analysis
│   ├── relationships.rs   # File-file co-edit patterns
│   └── types.rs
└── session/               # Session-level tracking
    └── mod.rs
```

### 3.5 Neuroscience modules (16)

| Module | Citation / basis | Purpose |
|---|---|---|
| `ActivationNetwork` | Collins & Loftus 1975 | Spreading activation across memory graph |
| `SynapticTaggingSystem` | Frey & Morris 1997 | Retroactive importance: memories in last 9h get boosted when big event fires |
| `HippocampalIndex` | Teyler & Rudy 2007 | Graph-level indexing; "dentate gyrus pattern separator" |
| `ContextMatcher` | Tulving 1973 | Encoding specificity — context overlap boosts retrieval by up to 30% |
| `AccessibilityCalculator` | Bjork & Bjork 1992 | `accessibility = retention × 0.5 + retrieval × 0.3 + storage × 0.2` |
| `CompetitionManager` | Anderson 1994 | Retrieval-induced forgetting — winners strengthen, competitors weaken |
| `StateUpdateService` | — | FSRS state transitions + append-only log |
| `ImportanceSignals` (4 channels) | Novelty / Arousal / Reward / Attention | Composite importance score, threshold 0.6 |
| `EmotionalMemory` | Brown & Kulik 1977 | Flashbulb memories — high-arousal events encode stronger |
| `PredictiveMemory` | Friston 2010 | Active inference — predict user needs before they ask |
| `ProspectiveMemory` | — | Intentions ("remind me when...") |
| `IntentionParser` | — | Natural-language → structured intention trigger |
| `MemoryState` (enum) | Bjork & Bjork 1992 | Active ≥0.7 / Dormant ≥0.4 / Silent ≥0.1 / Unavailable <0.1 |
| `Rac1Cascade` (v2.0.5) | Cervantes-Sandoval & Davis 2020 | Actin-destabilization-mediated forgetting of co-activated neighbors |
| `Suppression` (v2.0.5) | Anderson 2025 SIF | Top-down inhibitory control; compounds; 24h reversible labile window |
| `Reconsolidation` | Nader 2000 | 5-minute labile window after access; up to 10 modifications |

### 3.6 Advanced modules (11)

| Module | Purpose | Key methods |
|---|---|---|
| `ImportanceTracker` | Aggregates 4-channel score history | `record()`, `get_composite()` |
| `ReconsolidationManager` | Nader labile window state machine | `mark_labile()`, `apply_modification()`, `reconsolidate()` |
| `IntentDetector` | 9 intent types (Question, Decision, Plan, etc.) | `detect()` |
| `ActivityTracker` | Session-level active memory | `record_access()`, `recent()` |
| `MemoryDreamer` | 5-stage consolidation: Replay → Cross-reference → Strengthen → Prune → Transfer. Uses Waking SWR tagging (70% tagged + 30% random for diversity) | `dream(memory_count)` |
| `MemoryChainBuilder` | A*-like pathfinding between memories | `build_chain(from, to)` |
| `MemoryCompressor` | Semantic compression for 30+ day old memories | `compress(group)` |
| `CrossProjectLearner` | 6 pattern types (ErrorHandling, AsyncConcurrency, Testing, Architecture, Performance, Security) | `find_universal_patterns()`, `apply_to_project()` |
| `AdaptiveEmbedder` | Matryoshka-truncation tier selection | `embed_adaptive()` |
| `SpeculativeRetriever` | 6 trigger types for proactive memory fetch | `predict_needed_memories()` |
| `ConsolidationScheduler` | Runs FSRS decay cycle on interval (default 6h, env-configurable) | `start()` |

### 3.7 Storage

- SQLite via rusqlite 0.38, WAL mode, `Mutex<Connection>` split between reader and writer.
- FTS5 for keyword search (`bm25(10.0, 5.0, 1.0)` weights).
- Migrations V1..V11. **V11 (2026-04-19)** drops the dead `knowledge_edges` and `compressed_memories` tables that were reserved but never used.
- Append-only audit logs: `state_transitions`, `consolidation_history`, `dream_history`.

### 3.8 Embeddings

- Default: **Nomic Embed Text v1.5** via fastembed (ONNX, 768D).
- Matryoshka truncation to 256D for fast HNSW lookups (20× faster than full 768D).
- HyDE query expansion (generate a hypothetical document, embed it, search by its embedding).
- **v2.1 scaffolding:** Qwen3 embedding backend via Candle behind `qwen3-embed` feature. `qwen3_format_query()` helper prepends the instruction prefix ("Given a web search query, retrieve relevant passages that answer the query").
- Embedding cache: in-memory LRU; disk-warm on first run (~130MB for Nomic, ~500MB for Qwen3).

### 3.9 Vector search

- USearch HNSW (pinned 2.23.0; 2.24.0 regressed on MSVC per usearch#746). Int8 quantization.
- Hybrid scoring: `combined = 0.3 × BM25 + 0.7 × cosine` (default, user-tunable).
- `DEFAULT_DIMENSIONS` feature-gated: 256 on default, 1024 under `qwen3-embed`.

### 3.10 FSRS-6

- 21 trained parameters (Jarrett Ye / maimemo; trained on 700M+ Anki reviews).
- `R(t) = (1 + factor × t / S)^(-w20)` — power-law forgetting curve.
- 20-30% more efficient than SM-2 (Anki's original algorithm).
- Retrievability, stability, difficulty tracked per node.
- Dual-strength (Bjork & Bjork 1992): storage strength grows monotonically, retrieval strength decays.

### 3.11 Test count

- **364 `#[test]` annotations in vestige-core** across 47 test-bearing files.
- Examples: `cargo test --workspace` → 1,292 passing (includes 366 core + 425 mcp + e2e + journey).

---

## 4. `vestige-mcp` — MCP Server + Dashboard Backend

### 4.1 Purpose

Binary crate. Wraps `vestige-core` behind an MCP JSON-RPC 2.0 server, plus an embedded Axum HTTP server that hosts the dashboard, WebSocket event bus, and REST API.

### 4.2 Binaries

| Binary | Source | Purpose |
|---|---|---|
| `vestige-mcp` | `src/main.rs` | **Primary.** MCP JSON-RPC over stdio + optional HTTP transport. Hosts dashboard at `/dashboard/`. |
| `vestige` | `src/bin/cli.rs` | CLI: stats, consolidate, backup, restore, export, gc, dashboard launcher. |
| `vestige-restore` | `src/bin/restore.rs` | Standalone batch restore from JSON backup. |

### 4.3 Environment variables

| Var | Default | Purpose |
|---|---|---|
| `VESTIGE_DATA_DIR` | `~/.vestige/` | Storage root |
| `VESTIGE_DASHBOARD_PORT` | `3927` | Dashboard HTTP + WebSocket port |
| `VESTIGE_HTTP_PORT` | `3928` | Optional MCP-over-HTTP port |
| `VESTIGE_HTTP_BIND` | `127.0.0.1` | HTTP bind address |
| `VESTIGE_AUTH_TOKEN` | auto-generated | Dashboard bearer auth |
| `VESTIGE_CONSOLIDATION_INTERVAL_HOURS` | `6` | FSRS decay cycle cadence |
| `VESTIGE_DASHBOARD_ENABLED` | `true` | Toggle dashboard on/off |
| `VESTIGE_SYSTEM_PROMPT_MODE` | `minimal` | `full` enables the extended `build_instructions` block |
| `RUST_LOG` | `info` | tracing filter |

### 4.4 The 24 MCP tools

Every tool implemented in `src/tools/*.rs`. JSON schemas are programmatically emitted from `schema()` functions on each module.

1. **`session_context`** — one-call session init. Params: `queries[]`, `context{codebase, topics, file}`, `token_budget` (100-100000), `include_status`, `include_intentions`, `include_predictions`. Returns markdown context + `automationTriggers` (needsDream, needsBackup, needsGc) + `expandable` overflow IDs.
2. **`smart_ingest`** — single or batch ingest with Prediction Error Gating (similarity >0.92 → UPDATE, 0.75-0.92 → UPDATE/SUPERSEDE, <0.75 → CREATE). Params: `content`, `tags[]`, `node_type`, `source`, `forceCreate`, OR `items[]` (up to 20). Runs full cognitive pipeline.
3. **`search`** — 7-stage cognitive search. Params: `query`, `limit` (1-100), `min_retention`, `min_similarity`, `detail_level` (brief/summary/full), `context_topics[]`, `token_budget`, `retrieval_mode` (precise/balanced/exhaustive). **Strengthens retrieved memories via testing effect.**
4. **`memory`** — CRUD + promote/demote. `action` ∈ `{get, edit, delete, promote, demote, state, get_batch}`. `get_batch` takes up to 20 IDs. Edit preserves FSRS state, regenerates embedding.
5. **`codebase`** — Remember patterns & decisions. Actions: `remember_pattern`, `remember_decision`, `get_context`. Feeds CrossProjectLearner.
6. **`intention`** — Prospective memory. Actions: `set` (with trigger types time/context/event), `check`, `update`, `list`. Supports `include_snoozed` (v2.0.7 fix).
7. **`dream`** — 5-stage consolidation cycle. Param: `memory_count` (default 50). Returns insights, connections found, memories replayed, duration.
8. **`explore_connections`** — Graph traversal. Actions: `associations` (spreading activation), `chain` (A*-like path), `bridges` (connecting memories between two concepts).
9. **`predict`** — Proactive retrieval via SpeculativeRetriever. Param: `context{codebase, current_file, current_topics[]}`. Returns predictions with confidence + reasoning. Has a `predict_degraded` flag (v2.0.7) that surfaces warnings instead of silent empty responses.
10. **`importance_score`** — 4-channel scoring. Param: `content`, `context_topics[]`, `project`. Returns `{composite, channels{novelty, arousal, reward, attention}, recommendation}`.
11. **`find_duplicates`** — Cosine similarity clustering. Params: `similarity_threshold` (default 0.80), `limit`, `tags[]`. Returns merge/review suggestions.
12. **`memory_timeline`** — Chronological browse. Params: `start`, `end`, `node_type`, `tags[]`, `limit`, `detail_level`.
13. **`memory_changelog`** — Audit trail. Per-memory mode (by `memory_id`) or system-wide (with optional `start`/`end` ISO bounds, v2.0.7 fix adds 4× over-fetch when bounded).
14. **`memory_health`** — Retention dashboard. Returns avg retention, distribution buckets (0-20%, 20-40%, ...), trend, recommendation.
15. **`memory_graph`** — Visualization export. Params: `query` OR `center_id`, `depth` (default 2), `max_nodes` (default 50). Returns nodes with force-directed positions + edges with weights.
16. **`deep_reference`** — **★ THE killer tool.** 8-stage cognitive reasoning:
    1. Broad retrieval + cross-encoder reranking.
    2. Spreading activation expansion.
    3. FSRS-6 trust scoring (retention × stability × reps ÷ lapses).
    4. Intent classification (FactCheck / Timeline / RootCause / Comparison / Synthesis).
    5. Temporal supersession.
    6. Trust-weighted contradiction analysis.
    7. Relation assessment (Supports / Contradicts / Supersedes / Irrelevant).
    8. Template reasoning chain — pre-built natural-language conclusion the AI validates.
    Returns `{intent, reasoning, recommended, evidence, contradictions, superseded, evolution, related_insights, confidence}`.
17. **`cross_reference`** — Backward-compat alias that calls `deep_reference`. Kept for v1.x users.
18. **`system_status`** — Full health + stats + warnings + recommendations. Used by `session_context` when `include_status=true`.
19. **`consolidate`** — FSRS-6 decay cycle + embedding generation pass. Returns counts.
20. **`backup`** — SQLite backup to `~/.vestige/backups/` with timestamp.
21. **`export`** — JSON/JSONL export. Params: `format`, `tags[]`, `since`. v2.0.7 defensive `Err` on unknown format (was `unreachable!()`).
22. **`gc`** — Garbage collect. Params: `min_retention` (default 0.1), `dry_run` (default true). Dry-run first, then execute.
23. **`restore`** — Restore from backup. Param: `path`.
24. **`suppress`** / **`unsuppress`** (v2.0.5 "Intentional Amnesia") — Top-down inhibition. `suppress(id, reason?)` compounds (`suppressionCount` increments); `unsuppress(id)` reverses if within 24h labile window. Also exposed as dashboard HTTP endpoints (v2.0.7: `POST /api/memories/{id}/suppress` + `/unsuppress`).

### 4.5 MCP server internals

- `src/server.rs` — JSON-RPC 2.0 over stdio, optional HTTP. Handles `initialize`, `tools/list`, `tools/call`.
- **`build_instructions()`** — constructs the `instructions` string returned by `initialize`. Gated on `VESTIGE_SYSTEM_PROMPT_MODE=full`. Full mode emits an extended cognitive-protocol system prompt; default is concise.
- **CognitiveEngine** (`src/cognitive/mod.rs`) — async wrapper around `Arc<Storage>` + broadcast channel. Holds the WebSocket event sender.
- **Tool dispatch** — every `tools/call` invocation is routed to a `execute_*` function by tool name.

### 4.6 Dashboard HTTP backend (`src/dashboard/`)

- `src/dashboard/mod.rs` — Axum `Router` assembly.
- `src/dashboard/handlers.rs` — all REST handlers (~30 routes).
- `src/dashboard/static_files.rs` — embeds `apps/dashboard/build/` via `include_dir!` at compile time.
- `src/dashboard/state.rs` — `AppState { storage, event_tx, start_time }`.
- `src/dashboard/websocket.rs` — `/ws` upgrade handler with Origin validation (localhost + 127.0.0.1 + dev :5173), 64KB frame cap, 256KB message cap, heartbeat task every 5s.

**Heartbeat payload (v2.0.7):** `{type: "Heartbeat", data: {uptime_secs, memory_count, avg_retention, suppressed_count, timestamp}}`. The `uptime_secs` is what powers the sidebar footer's `formatUptime()` display ("3d 4h" / "18m 43s").

### 4.7 WebSocket event bus — 19 VestigeEvent types

Emitted from the `CognitiveEngine` broadcast channel to every connected dashboard client:

| Event | When emitted | Dashboard visual |
|---|---|---|
| `Connected` | WebSocket upgrade complete | Cyan ripple (v2.0.6) |
| `Heartbeat` | Every 5s | Silent (updates sidebar stats) |
| `MemoryCreated` | Any ingest that produces a new node | Rainbow burst + double shockwave + ripple |
| `MemoryUpdated` | Smart_ingest UPDATE path | Pulse at node |
| `MemoryDeleted` | `memory({action: "delete"})` | Dissolution animation |
| `MemoryPromoted` | `memory({action: "promote"})` | Green pulse + sparkle |
| `MemoryDemoted` | `memory({action: "demote"})` | Orange pulse + fade |
| `MemorySuppressed` | `suppress(id)` (v2.0.5) | Violet implosion (v2.0.7) |
| `MemoryUnsuppressed` | `unsuppress(id)` (v2.0.5) | Rainbow reversal (v2.0.7) |
| `Rac1CascadeSwept` | Rac1 worker completes cascade (72h async) | Violet wave pulse (v2.0.6) |
| `SearchPerformed` | Every `search()` call | Cyan flash + PipelineVisualizer 7-stage animation in `/feed` |
| `DreamStarted` | `dream()` begins | Scene enters dream mode (2s lerp) |
| `DreamProgress` | Per-stage updates during dream | Aurora hue cycle |
| `DreamCompleted` | Dream finishes, insights generated | Scene exits dream mode |
| `ConsolidationStarted` | FSRS consolidation cycle starts | Amber warning pulse (v2.0.6) |
| `ConsolidationCompleted` | Consolidation finishes | Green confirmation pulse |
| `RetentionDecayed` | Node's retention drops below threshold during consolidation | Red decay pulse |
| `ConnectionDiscovered` | Dream or spreading activation finds new edge | **Cyan flash on edge (already fires — NOT yet surfaced as a toast; see v2.2 "Pulse")** |
| `ActivationSpread` | Spreading activation from a memory | Turquoise ripple (v2.0.6) |
| `ImportanceScored` | `importance_score()` or internal scoring event | Hot-pink pulse (v2.0.6, magenta) |

### 4.8 Dashboard REST API

All routes under `/api/`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check (status, version, memory count) |
| GET | `/api/stats` | Full stats (same surface as `system_status` tool) |
| GET | `/api/memories` | List memories with filters (q, node_type, tag, min_retention) |
| GET | `/api/memories/{id}` | Single memory detail |
| POST | `/api/memories` | Create memory (raw ingest) |
| DELETE | `/api/memories/{id}` | Delete |
| POST | `/api/memories/{id}/promote` | Promote (+0.20 retrieval, +0.10 retention, 1.5× stability) |
| POST | `/api/memories/{id}/demote` | Demote (−0.30 retrieval, −0.15 retention, 0.5× stability) |
| POST | `/api/memories/{id}/suppress` | v2.0.7: compound suppression |
| POST | `/api/memories/{id}/unsuppress` | v2.0.7: reverse within 24h labile window |
| POST | `/api/search` | Hybrid search (keyword + semantic weights) |
| POST | `/api/ingest` | Smart ingest (PE gating) |
| GET | `/api/graph` | Graph visualization export |
| POST | `/api/explore` | Actions: associations / chains / bridges |
| POST | `/api/dream` | Run dream cycle |
| POST | `/api/consolidate` | Run FSRS decay cycle |
| POST | `/api/predict` | Proactive predictions |
| POST | `/api/importance` | 4-channel score |
| GET | `/api/timeline` | Chronological |
| GET | `/api/intentions` | List intentions (filter by status) |
| GET | `/api/retention-distribution` | Bucketed histogram |

WebSocket: `GET /ws` (upgrade) — one broadcast channel, any connected client gets all events.

### 4.9 vestige-mcp feature flags

| Flag | Purpose | Default |
|---|---|---|
| `embeddings` | Forward to vestige-core | yes |
| `vector-search` | Forward to vestige-core | yes |
| `ort-dynamic` | Forward to vestige-core | no |

Build commands (from CONTRIBUTING.md):
- Full: `cargo install --path crates/vestige-mcp`
- No-embeddings (tiny): `cargo install --path crates/vestige-mcp --no-default-features`
- Dynamic ORT (glibc < 2.38): `cargo install --path crates/vestige-mcp --no-default-features --features ort-dynamic,vector-search`

---

## 5. `apps/dashboard` — SvelteKit + Three.js Frontend

### 5.1 Purpose

Interactive 3D graph + CRUD + analytics dashboard. Built with SvelteKit 2 + Svelte 5 runes, embedded into the Rust binary via `include_dir!` and served at `/dashboard/`.

### 5.2 Tech stack

- **SvelteKit 2.53** + **Svelte 5.53** (runes: `$state`, `$props`, `$derived`, `$effect`).
- **Three.js 0.172** — WebGL, MSAA, ACESFilmic tone mapping.
- **Tailwind CSS 4.2** — custom `@theme` block (synapse, dream, memory, recall, decay colors + 8 node-type palette).
- **TypeScript 5.9** — strict mode.
- **Vite 6.4** + **Vitest 4.0.18** (251 tests).
- **@playwright/test 1.58** — E2E ready (journeys live in `tests/vestige-e2e-tests/`).

### 5.3 Routes (SvelteKit file-based)

Grouped under `(app)/`:

| Route | File | Purpose |
|---|---|---|
| `/` | `+page.svelte` | Redirect to `/graph` |
| `(app)/graph` | `+page.svelte` | **Primary 3D graph** (Graph3D component + color mode toggle + time slider + right panel for detail + legend overlay v2.0.8) |
| `(app)/memories` | `+page.svelte` | Memory browser (search, filter by type/tag/retention, suppress button v2.0.7) |
| `(app)/intentions` | `+page.svelte` | Prospective memory + predictions (status tabs, trigger icons, priority labels) |
| `(app)/stats` | `+page.svelte` | Health dashboard, retention distribution, endangered memories, run-consolidation button |
| `(app)/timeline` | `+page.svelte` | Chronological browse (days dropdown, expandable day cards) |
| `(app)/feed` | `+page.svelte` | Live event stream (200-event FIFO buffer, PipelineVisualizer on SearchPerformed) |
| `(app)/explore` | `+page.svelte` | Associations / Chains / Bridges mode toggle + Importance Scorer |
| `(app)/settings` | `+page.svelte` | Operations + config + keyboard shortcuts reference |

### 5.4 Root layout (`src/routes/+layout.svelte`)

- Desktop sidebar (8 nav items) + mobile bottom nav (5 items).
- **Command palette (⌘K)** — opens a search bar that navigates.
- **Single-key shortcuts** — G/M/T/F/E/I/S for routes.
- **Status footer** — connection indicator, memory count, avg retention, suppressed count (v2.0.5), uptime (v2.0.7: `up {formatUptime($uptimeSeconds)}`).
- **ForgettingIndicator** — violet badge showing suppressed count.
- Ambient orb background animations (CSS).

### 5.5 Components (`src/lib/components/`)

| Component | Purpose |
|---|---|
| `Graph3D.svelte` | **The 3D canvas.** Props: `nodes[]`, `edges[]`, `centerId`, `events[]`, `isDreaming`, `colorMode` (v2.0.8), `onSelect`, `onGraphMutation`. Owns the Three.js scene and all module init. |
| `MemoryStateLegend.svelte` (v2.0.8) | Floating overlay explaining 4 FSRS buckets — only renders when `colorMode === 'state'`. |
| `PipelineVisualizer.svelte` | 7-stage cognitive search animation (Overfetch → Rerank → Temporal → Access → Context → Compete → Activate). Shown in `/feed` when SearchPerformed arrives. |
| `RetentionCurve.svelte` | SVG FSRS-6 decay curve in the graph right panel. `R(t) = e^(-t/S)` with predictions at Now / 1d / 7d / 30d. |
| `TimeSlider.svelte` | Temporal playback scrubber. State: enabled, playing, speed (0.5-2×), sliderValue. Callbacks `onDateChange`, `onToggle`. |
| `ForgettingIndicator.svelte` | Violet badge in sidebar showing suppressed count from Heartbeat. |

### 5.6 Three.js graph system (`src/lib/graph/`)

| File | Role |
|---|---|
| `nodes.ts` | `NodeManager`. Fibonacci sphere initial positions, materialize/dissolve/grow animations, shared radial-gradient glow texture (128px) that prevents square bloom artifacts (issue #31). **v2.0.8:** `ColorMode` ('type' / 'state'), `getMemoryState(retention)`, `MEMORY_STATE_COLORS`, `MEMORY_STATE_DESCRIPTIONS`, `setColorMode(mode)` idempotent in-place retint. **2026-04-19:** dark-glass-pill label redesign (dimmer `#94a3b8` slate on `rgba(10,16,28,0.82)` pill with hairline stroke). |
| `edges.ts` | `EdgeManager`. Violet `#8b5cf6` lines; opacity = 25% + 50% × weight, capped at 80%. Grow/dissolve animations. |
| `force-sim.ts` | Repulsion 500, attraction 0.01 × edge weight × distance, damping 0.9, centering 0.001α. N² but fine up to ~1000 nodes at 60fps. |
| `particles.ts` | `ParticleSystem`. Starfield (3000 points on spherical shell r=600-1000) + neural particles (500 oscillating sin-wave). |
| `effects.ts` | `EffectManager`. 12 effect types (SpawnBurst, Shockwave, RainbowBurst, RippleWave, Implosion, Pulse, ConnectionFlash, etc.). |
| `events.ts` | `mapEventToEffects()` — maps every one of the 19 VestigeEvent variants to a visual effect. Live-spawn mechanics: new nodes spawn near semantically related existing nodes (tag + type scoring), FIFO eviction at 50 nodes. |
| `scene.ts` | Scene factory. Camera 60° FOV at (0, 30, 80). ACESFilmic tone mapping, exposure 1.25, pixel ratio clamped ≤2×. **UnrealBloomPass:** strength 0.55, radius 0.6, threshold 0.2 (retuned v2.0.8 for radial-gradient sprites). OrbitControls with auto-rotate 0.3°/frame. |
| `dream-mode.ts` | Smooth 2s lerp between NORMAL (bloom 0.8, rotate 0.3, fog dense) and DREAM (bloom 1.8, rotate 0.08, nebula 1.0, chromatic 0.005). Aurora lights cycle hue in dream. |
| `temporal.ts` | `filterByDate(nodes, edges, cutoff)`, `retentionAtDate(current, stability, created, target)` using FSRS decay formula. Enables the TimeSlider preview. |
| `shaders/nebula.frag.ts` | Nebula background fragment shader (purple → cyan → magenta cycle with turbulence). |
| `shaders/post-processing.ts` | Chromatic aberration, vignette, subtle distortion. Parameters lerp with dream-mode. |

### 5.7 Stores (`src/lib/stores/`)

| Store | Exports | Purpose |
|---|---|---|
| `api.ts` | `api.memories.*`, `api.search`, `api.graph`, `api.explore`, `api.stats`, `api.health`, `api.retentionDistribution`, `api.timeline`, `api.dream`, `api.consolidate`, `api.predict`, `api.importance`, `api.intentions` | 23 REST client methods |
| `websocket.ts` | `websocket` (writable), `isConnected`, `eventFeed`, `heartbeat`, `memoryCount`, `avgRetention`, `suppressedCount`, `uptimeSeconds`, `formatUptime(secs)` | WebSocket connection + derived state. FIFO 200-event ring buffer. Exponential backoff reconnect (1s → 30s). |
| `graph-state.svelte.ts` | (unused artifact from v2.0.6) | — |

### 5.8 Types (`src/lib/types/index.ts`)

Exported: `Memory`, `SearchResult`, `MemoryListResponse`, `SystemStats`, `HealthCheck`, `RetentionDistribution`, `GraphNode`, `GraphEdge`, `GraphResponse`, `DreamResult`, `DreamInsight`, `ImportanceScore`, `ConsolidationResult`, `SuppressResult`, `UnsuppressResult`, `IntentionItem`, `VestigeEventType`, `VestigeEvent`, `NODE_TYPE_COLORS` (8 types), `EVENT_TYPE_COLORS` (19 events), `ColorMode`, `MemoryState` (v2.0.8).

### 5.9 Tests (`src/lib/graph/__tests__/`)

| File | Tests | Lines | Covers |
|---|---|---|---|
| `color-mode.test.ts` **(v2.0.8, new)** | 80 | 664 | `getMemoryState` boundaries (12 retentions including NaN/±∞/>1/<0), palette integrity, `getNodeColor` dispatch, `NodeManager.setColorMode` idempotence + in-place retint + userData preservation + suppression channel isolation |
| `nodes.test.ts` | 32 | 456 | NodeManager lifecycle, easings, Fibonacci distribution |
| `edges.test.ts` | 21 | 314 | EdgeManager grow/dissolve, opacity-by-weight |
| `force-sim.test.ts` | 19 | 257 | Physics convergence, add/remove |
| `effects.test.ts` | 30 | 500 | All 12 effect types |
| `events.test.ts` | 48 | 864 | Every one of the 19 event handlers + live-spawn + eviction |
| `ui-fixes.test.ts` | 21 | 236 | Bloom retuning, glow-texture gradient, fog density, regression tests for issue #31 |
| **Total** | **251** | **3,291** | |

Infrastructure: `three-mock.ts` (Scene / Mesh / Sprite / Material mocks), `setup.ts` (canvas context mocks including `beginPath`/`closePath`/`quadraticCurveTo` added tonight for the pill redesign), `helpers.ts` (node/edge/event factories).

### 5.10 Build

- `pnpm run build` → static SPA in `apps/dashboard/build/`.
- Precompressed `.br` + `.gz` per asset (adapter-static).
- **Embedded into `vestige-mcp` binary** at compile time via `include_dir!("$CARGO_MANIFEST_DIR/../../apps/dashboard/build")`. Every Rust build rebakes the dashboard snapshot.

---

## 6. Integrations & Packaging

### 6.1 IDE integration matrix (`docs/integrations/*.md`)

All 8 IDEs documented. The common install flow: (a) download `vestige-mcp` binary, (b) point IDE's MCP config at its absolute path, (c) restart IDE, (d) verify with `/context` or equivalent.

| IDE | Config path | Notable |
|---|---|---|
| Claude Code | `~/.claude.json` or project `.mcp.json` | Inline in `CONFIGURATION.md`; one-liner install |
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` | Inline in `CONFIGURATION.md` |
| Cursor | `~/.cursor/mcp.json` | Absolute paths required (Cursor doesn't resolve relatives reliably) |
| VS Code (Copilot) | `.vscode/mcp.json` OR User via command | **Uses `"servers"` key, NOT `"mcpServers"`** — Copilot-specific schema. Requires agent mode enabled. |
| Codex | `~/.codex/config.toml` | TOML not JSON. `codex mcp add vestige -- /usr/local/bin/vestige-mcp` helper. |
| Xcode | Project-level `.mcp.json` | **Xcode 26.3's `claudeai-mcp` feature gate blocks global config. Project-level `.mcp.json` in project root bypasses entirely.** First cognitive memory server for Xcode. Sandboxed agents do NOT inherit shell env — absolute paths mandatory. |
| JetBrains / Junie | `.junie/mcp/mcp.json` or UI config | 2025.2+. Three paths: Junie autoconfig, Junie AI config, external MCP client. |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | Supports `${env:HOME}` variable expansion. Cascade AI. |

### 6.2 npm packages

| Package | Version | Role |
|---|---|---|
| `vestige-mcp-server` (in `packages/vestige-mcp-npm`) | 2.0.7 | Binary wrapper — postinstall downloads the platform-appropriate release asset from GitHub. Bins: `vestige-mcp`, `vestige`. |
| `@vestige/init` (in `packages/vestige-init`) | 2.0.7 | Interactive zero-config installer. Bin: `vestige-init`. |
| `packages/vestige-mcpb/` | — | Legacy, abandoned. |

**Publish status:** v2.0.6 is live on npm. **v2.0.7 pending Sam's Touch ID** (WebAuthn 2FA flow, not TOTP — has to be triggered from Sam's machine).

### 6.3 GitHub release workflow (`release.yml`)

Triggered on tag push (`v*`) OR manual `workflow_dispatch`. Matrix:

| Target | Runner | Artifact | Status |
|---|---|---|---|
| `aarch64-apple-darwin` | macos-latest | `vestige-mcp-aarch64-apple-darwin.tar.gz` | ✅ |
| `x86_64-unknown-linux-gnu` | ubuntu-latest | `vestige-mcp-x86_64-unknown-linux-gnu.tar.gz` | ✅ |
| `x86_64-pc-windows-msvc` | windows-latest | `vestige-mcp-x86_64-pc-windows-msvc.zip` | ✅ |
| `x86_64-apple-darwin` (Intel Mac) | **DROPPED in v2.0.7** | — | ❌ `ort-sys 2.0.0-rc.11` (pinned by fastembed 5.13.2) has no Intel Mac prebuilt |

Each artifact contains three binaries: `vestige-mcp`, `vestige`, `vestige-restore`.

### 6.4 CI workflow (`ci.yml`)

Triggers: push main + PR main. Runs on macos-latest + ubuntu-latest. Steps: `cargo check` → `cargo clippy --workspace -- -D warnings` → `cargo test --workspace`. **Tonight's fix:** Rust 1.95 introduced `unnecessary_sort_by` (12 sites fixed) + `collapsible_match` (1 site fixed in `memory_states.rs`, 1 `#[allow]` on `websocket.rs` because match guards can't move non-Copy `Bytes`).

### 6.5 Test workflow (`test.yml`)

5 parallel jobs: `unit-tests`, `mcp-tests`, `journey-tests` (depends on unit), `dashboard` (pnpm + vitest), `coverage` (LLVM + Codecov). Env: `VESTIGE_TEST_MOCK_EMBEDDINGS=1` to skip ONNX model download in CI.

### 6.6 Xcode setup script (`scripts/xcode-setup.sh`)

4.9 KB interactive installer. (a) detect/install binary, (b) offer project picker under `~/Developer`, (c) generate `.mcp.json`, (d) optionally batch-install to all detected projects. Supports SHA-256 checksum verification.

---

## 7. `vestige-cloud` — Current Skeleton

**Location:** `/Users/entity002/Developer/vestige-cloud` (separate git repo, private).

**Status as of 2026-04-19:** single-commit skeleton from 2026-02-12 (8 weeks old, one feature commit `4e181a6`). ~600 LOC.

### 7.1 Structure

```
vestige-cloud/
├── Cargo.toml                       # workspace, path-dep on ../vestige/crates/vestige-core
├── Cargo.lock
└── crates/
    └── vestige-http/
        ├── Cargo.toml               # binary: vestige-http
        └── src/
            ├── main.rs              # Axum server on :3927, auth + cors middleware
            ├── auth.rs              # Single bearer token via VESTIGE_API_KEY env, open if unset
            ├── cors.rs              # prod: allowlist vestige.dev + app.vestige.dev; dev: permissive
            ├── state.rs             # Arc<Mutex<Storage>> shared state (SINGLE TENANT)
            ├── sse.rs               # /mcp/sse STUB — 3 TODOs, returns one static "endpoint" event
            └── handlers/
                ├── mod.rs
                ├── health.rs        # GET /health (version + memory count)
                ├── api.rs           # REST CRUD: search, list, create, get, delete, promote, demote, stats, smart_ingest
                ├── mcp.rs           # POST /mcp JSON-RPC 2.0 — **ONLY 5 TOOLS** (search, smart_ingest, memory, promote_memory, demote_memory)
                └── sync.rs          # POST /sync/push + /sync/pull (sync/pull has TODO for `since` filter)
```

### 7.2 Gap analysis vs. current `vestige-mcp`

| Dimension | vestige-mcp v2.0.7 | vestige-cloud Feb skeleton | Gap |
|---|---|---|---|
| MCP tools | 24 | 5 | 19 tools missing (session_context, dream, explore_connections, predict, importance_score, find_duplicates, memory_timeline, memory_changelog, memory_health, memory_graph, deep_reference, consolidate, backup, export, gc, restore, intention, codebase, suppress/unsuppress) |
| MCP transport | stdio + HTTP | HTTP only, no Streamable HTTP | Needs full Streamable HTTP (`Mcp-Session-Id` header, bidirectional, Last-Event-ID reconnect) per 2025-06-18 spec |
| Multi-tenancy | N/A (local) | **Single tenant** (one storage, one API key) | Need per-user DB, row-level scoping, or DB-per-tenant sharding |
| Auth | Local token | Single bearer | Need JWT, OAuth, scopes, org membership, token rotation |
| Billing | N/A | none | Need Stripe, entitlement, plans, webhooks |
| Observability | `tracing` only | `tracing` only | Need Prometheus / OTLP export, dashboards, rate limits, error budget |
| Sync | N/A | lossy push + unfiltered pull | Need tombstones, incremental pull by `since`, conflict resolution |
| Deploy | binaries + npm | **none** | Need Dockerfile, fly.toml, CI, docs |

### 7.3 Two upgrade paths

- **Path A (v2.6.0 "Remote"):** Upgrade the Feb skeleton to match v2.0.7 surface (5 → 24 tools), implement Streamable HTTP, ship Dockerfile + fly.toml. **Keep single-tenant.** Ship as "deploy your own Vestige on a VPS."
- **Path B (v3.0.0 "Cloud"):** Multi-tenant SaaS. Weeks of work on billing, per-tenant DB, ops. Not viable until v2.6 has traction + cashflow.

The recommendation in Part 9 is **A only** for now. B is gated on demand signal + runway.

---

## 8. Version History (v1.0 → v2.0.8)

### 8.1 Shipped releases

| Version | Tag | Date | Theme | Headline |
|---|---|---|---|---|
| v1.0.0 | v1.0.0 | 2026-01-25 | Initial | First MCP server with FSRS-6 memory |
| v1.1.x | v1.1.0/1/2 | — | CLI separation | stats/health moved out of MCP to CLI |
| v1.3.0 | v1.3.0 | — | — | Importance scoring, session checkpoints, duplicate detection |
| v1.5.0 | v1.5.0 | — | — | Cognitive engine, memory dreaming, graph exploration, predictive retrieval |
| v1.6.0 | v1.6.0 | — | — | 6× storage reduction, neural reranking, instant startup |
| v1.7.0 | v1.7.0 | — | — | 18 tools, automation triggers, SQLite perf |
| v1.9.1 | v1.9.1 | — | Autonomic | Self-regulating memory, graph visualization |
| **v2.0.0** | v2.0.0 | **2026-02-22** | "Cognitive Leap" | 3D SvelteKit+Three.js dashboard, WebSocket event bus (16 events), HyDE query expansion, Nomic v2 MoE option, Command palette, bloom post-processing |
| v2.0.1 | v2.0.1 | — | — | Release rebuild, install fixes |
| v2.0.3 | v2.0.3 | — | — | Clippy fixes, CI alignment |
| v2.0.4 | v2.0.4 | 2026-04-09 | "Deep Reference" | **8-stage cognitive reasoning tool, `cross_reference` alias**, retrieval_mode (precise/balanced/exhaustive), token budgets raised 10K → 100K, CORS hardening |
| v2.0.5 | v2.0.5 | 2026-04-14 | "Intentional Amnesia" | **Active forgetting** — suppress tool #24, Rac1 cascade (72h async neighbour decay), 24h labile reversal window, graph node visual suppression (20% opacity, no emissive) |
| v2.0.6 | v2.0.6 | 2026-04-18 | "Composer" | 6 live graph reactions (Suppressed, Unsuppressed, Rac1, Connected, ConsolidationStarted, ImportanceScored), `VESTIGE_SYSTEM_PROMPT_MODE=full` opt-in |
| **v2.0.7** | v2.0.7 | 2026-04-19 | "Visible" | V11 migration drops dead tables; `/api/memories/{id}/suppress` + `/unsuppress` endpoints + UI button; sidebar `up 3d 4h` footer via `uptime_secs`; graph error-state split; `predict` degraded flag; `changelog` start/end honored; `intention` include_snoozed; `suppress` MCP tool (was dashboard-only); tool-count reconciled 23 → 24; Intel Mac dropped from release workflow; defensive `Err` on unknown export format |
| **v2.0.8** | *(unreleased, merged to main 2026-04-19 22:10 CT)* | — | — | FSRS memory-state colour mode (`ColorMode` type/state toggle) + floating legend + dark-glass-pill label redesign + 80 new tests + Rust 1.95 clippy compat (12 sites) |

### 8.2 Current git state

- **HEAD:** `main` at `30d92b5` "feat(graph): redesign node labels as dark glass pills"
- **Last 4 commits on main (v2.0.8):**
  - `30d92b5` — Label pill redesign
  - `d7f0fe0` — 80 new color-mode tests
  - `318d4db` — Rust 1.95 clippy compat
  - `4c20165` — Memory-state color mode + legend
- **Branches:**
  - `main` (default, protected via CI-must-pass)
  - `feat/v2.0.8-memory-state-colors` (fast-forwarded into main tonight)
  - `feat/v2.1.0-qwen3-embed` (Day 2 done; Day 3 pending on Sam's M3 Max arrival)
  - `chore/v2.0.7-clean` (post-v2.0.7 cleanup branch)
  - `wip/v2.0.7-v11-migration` (transport branch for cross-machine stash)
- **Latest tag:** `v2.0.7` (force-updated on main after v2.0.6 rebase incident)
- **Latest CI run on main:** #24646176395 ✅ all 4 jobs (Test macos, Test ubuntu, Release aarch64-darwin, Release x86_64-linux)

### 8.3 Open GitHub issues / PRs

- **Closed #35** — "npm publish delay 2.0.6"; replied in v2.0.6 with one-liner install command
- **Open #36** — desaiuditd: "hooks-for-automatic-memory request" — customer conversion opportunity, not yet responded

---

## 9. The Next-Phase Plan

**Shipping cadence:** weekly minor bumps (v2.1 → v2.2 → v2.3 ...) until v3.0 which gates on multi-tenancy + CoW storage. Ships ~Monday each week with content post same day + follow-up Wednesday + YouTube Friday.

### 9.1 v2.1.0 "Decide" — Qwen3 embeddings *(in-flight)*

**Branch:** `feat/v2.1.0-qwen3-embed` (pushed).
**Status:** scaffolding merged; Day 3 pending.
**ETA:** ~1 week after M3 Max arrival (FedEx hold at Walgreens, pickup 2026-04-20).

**What's in:** `qwen3-embed` feature flag gates a Candle-based Qwen3 embed backend. `qwen3_format_query()` helper for the query-instruction prefix. Metal device selection with CPU fallback. `DEFAULT_DIMENSIONS` feature-gated 256/1024. Dual-index routing scaffolded.

**What's left (Day 3):**
- Storage write-path records `embedding_model` per node.
- `semantic_search_raw` uses `qwen3_format_query` when feature active.
- Dual-index routing: old Nomic-256 nodes stay on their HNSW, new Qwen3-1024 nodes go on a new HNSW. Search merges with trust weighting.
- End-to-end test: ingest on Qwen3 → retrieve on Qwen3 at higher accuracy than Nomic.

**Test gate:** `cargo test --workspace --features qwen3-embed --release` green. Current baseline: 366 core + 425 mcp passing.

### 9.2 v2.2.0 "Pulse" — Subconscious Cross-Pollination **★ VIRAL LOAD-BEARING RELEASE**

**ETA:** 1-2 weeks after v2.1 lands.

**What it does:** While the user is doing anything else (typing a blog post, looking at a different tab, doing nothing), Vestige's running `dream()` in the background. When dream completes with `insights_generated > 0` or a `ConnectionDiscovered` event fires from spreading activation, **the dashboard pulses a toast** on the side: *"Vestige found a connection between X and Y. Here's the synthesis."* The bridging edge in the 3D graph flashes cyan and briefly thickens.

**Why viral:** This is the single most tweet/YouTube-friendly demo in the entire roadmap. It is the "my 3D brain is thinking for itself" moment.

**Backend (≈2 days):**
1. `ConsolidationScheduler` gains a "pulse" hook: after each cycle, if `insights_generated > 0` emit a new `InsightSurfaced` event with `{source_memory_id, target_memory_id, synthesis_text, confidence}`.
2. The existing `ConnectionDiscovered` event gets a richer payload: include both endpoint IDs + a templated synthesis string derived from the two memories' content.
3. Rate-limit pulses: max 1 per 15 min unless user is actively using the dashboard.

**Frontend (≈5 days):**
1. New Svelte component `InsightToast.svelte` — slides in from right, shows synthesis text + "View connection" button, auto-dismisses after 10s.
2. `events.ts` mapping: `InsightSurfaced` → locate bridging edge in graph, pulse it cyan for 2s, thicken to 2× for 500ms, play a soft chime (optional, muted by default).
3. Toast queue so rapid dreams don't flood.
4. Preference: user can toggle pulse sound / toast / edge animation independently in `/settings`.

**Already exists (nothing to build):**
- `dream()` 5-stage cycle — YES
- `DreamCompleted` event with `insights_generated` — YES
- `ConnectionDiscovered` event + WebSocket broadcast — YES
- 3D edge animation system in `events.ts` — YES (handler exists, just doesn't emit toast)
- ConsolidationScheduler running on `VESTIGE_CONSOLIDATION_INTERVAL_HOURS` — YES

**Never-composed alarm:** Four existing components, zero lines of composition. This feature is **~90% latent in v2.0.7**. All we do is press the button.

**Acceptance criteria:**
- Start Vestige, idle for 10 min, verify a pulse fires from scheduled dream cycle.
- Ingest 3 semantically adjacent memories from completely different domains (e.g., F1 aerodynamics, memory leak, fluid dynamics), trigger dream, verify connection pulse fires with synthesis text mentioning both source + target.
- Dashboard test coverage: add `pulse.test.ts` with 15+ cases covering toast queue, rate limit, event shape, edge animation.

**Launch day:** Film a 90-second screen recording. Post to Twitter + Hacker News + LinkedIn + YouTube same day.

### 9.3 v2.3.0 "Rewind" — Time Machine

**ETA:** 2-3 weeks after v2.2 ships.

**What it does:** The graph page gets a horizontal time slider. Drag back in time → nodes dim based on retroactive FSRS retention, edges that were created after the slider's timestamp dissolve visibly, suppressed memories un-dim to their pre-suppression state. A "Pin" button snapshots the current slider state into a named checkpoint the user can return to.

**Backend (≈4 days):**
1. New core API: `Storage::memory_state_at(memory_id, timestamp) -> MemorySnapshot` — reconstructs a node's FSRS state at an arbitrary past timestamp by replaying `state_transitions` forward OR applying FSRS decay backward from the current state.
2. New MCP tool: `memory_graph_at(query, depth, max_nodes, timestamp)` — the existing graph call with a time parameter.
3. New MCP tool: `pin_state(name, timestamp)` — persists a named snapshot (just a row in a new `pins` table: name, timestamp, created_at).
4. New core API: `list_pins()` + `delete_pin(name)`.

**Frontend (≈7 days):**
1. `TimeSlider.svelte` already exists as a scaffold (listed in §5.5) — upgrade it to an HTML5 range input + play/pause + speed control.
2. Graph3D consumes a new `asOfTimestamp` prop. When set, uses `temporal.ts::retentionAtDate()` to re-project every node's opacity + size.
3. Edges: hide those with `created_at > slider`. Animate the dissolution so sliding feels organic.
4. Pin sidebar: list pinned states, click to jump, rename/delete.

**Cut from scope: branching.** Git-like "what if I forgot my Python biases" requires CoW storage = full schema migration = v3.0 territory. Scope it out explicitly.

**Acceptance criteria:**
- Slide back 30 days, verify node count drops to whatever existed 30 days ago.
- Slide back through a suppression event, verify node un-dims.
- Pin "before Mays deadline", verify pin jumps restore exact state.

### 9.4 v2.4.0 "Empathy" — Emotional Context Tagging **★ FIRST PRO-TIER GATE CANDIDATE**

**ETA:** 2-3 weeks after v2.3 ships.

**What it does:** Vestige's MCP middleware watches tool call metadata for frustration signals — repeated retries of the same query, CAPS LOCK content, explicit correction phrases ("no that's wrong", "actually..."), rapid-fire consecutive calls. When detected, the current active memory gets an automatic `ArousalSignal` boost and a `frustration_detected_at` timestamp. Next session, when the user returns to a similar topic, the agent proactively surfaces: *"Last time we worked on this, you were frustrated with the API docs. I've pre-read them."*

**Why Pro-tier:** Invisible to demo (so doesn't hurt OSS growth), creates deep lock-in, quantifiable value ("Vestige saved you X minutes of re-frustration this month"), clear paid-hook rationale.

**Backend (≈4 days):**
1. New middleware layer in `vestige-mcp` between JSON-RPC dispatch and tool execution: `FrustrationDetector`. Analyzes tool args for: (a) retry pattern (same `query` field within 60s), (b) content ≥70% caps after lowercase comparison, (c) correction regex (`no\s+that|actually|wrong|fix this|try again`).
2. On detection, fire a synthesized `ArousalSignal` to `ImportanceTracker` for the most-recently-accessed memory.
3. New core API: `find_frustration_hotspots(topic, limit)` → returns memories with `arousal_score > threshold` + their `frustration_detected_at` timestamps.
4. `session_context` tool gains a new field: `frustration_warnings[]` — "Topic X had previous frustration; here's what we know."

**Frontend (≈3 days):**
1. Memory detail pane shows an orange "Frustration" badge for high-arousal memories.
2. `/stats` adds a "Frustration hotspots" section.

**Acceptance criteria:**
- Simulate 3 rapid retries of the same query, verify ArousalSignal boosts the active memory.
- Simulate caps-lock content, verify detection.
- Return to same topic next session, verify `session_context` surfaces warning.

### 9.5 v2.5.0 "Grip" — Neuro-Feedback Cluster Gestures

**ETA:** 2 weeks after v2.4 ships.

**What it does:** In the 3D graph, drag a memory sphere to "grab" it — its cluster highlights. Squeeze (pinch gesture or modifier key + drag inward) → promotes the whole cluster. Flick away (throw gesture) → triggers decay on the cluster.

**Backend (≈2 days):**
1. New MCP tool: `promote_cluster(memory_ids[])` — applies promote to each.
2. New MCP tool: `demote_cluster(memory_ids[])` — inverse.
3. Cluster detection helper: `find_cluster(source_id, similarity_threshold)` — leverages existing `find_duplicates` + spreading activation.

**Frontend (≈5 days):**
1. Three.js gesture system: drag detection, cluster highlight (emissive pulse on all cluster members), squeeze detection (pointer velocity inward), flick detection (pointer velocity outward past threshold).
2. Visual feedback: green ring on squeeze (promote), red dissipation on flick (demote).
3. Accessibility: keyboard alternative — select node, press `P` / `D` to promote/demote cluster.

### 9.6 v2.6.0 "Remote" — `vestige-cloud` Self-Host Upgrade

**ETA:** 3 weeks after v2.5 ships. First paid-tier candidate if empathy doesn't convert first.

**What it does:** Turns the Feb `vestige-cloud` skeleton into a shippable self-host product. One-liner install → Docker container or fly.io deploy → point Claude Desktop/Cursor/Codex at the remote URL → cloud-persistent memory across all your devices.

**Scope:**
1. Upgrade MCP handler from 5 → 24 tools (port each tool from `crates/vestige-mcp/src/tools/`).
2. Implement **MCP Streamable HTTP transport** (spec 2025-06-18): `Mcp-Session-Id` header, bidirectional event stream, Last-Event-ID reconnect, JSON-RPC batching.
3. Per-user SQLite at `/data/$USER_ID.db` (single-tenant but scoped by `VESTIGE_USER_ID` env — "single-tenant but deploy-multiple").
4. `Dockerfile` (multi-stage: Rust build + fastembed model baked in).
5. `fly.toml` with persistent volume mount on `/data`.
6. `docker-compose.yml` for local Postgres-if-needed (probably not — stick with SQLite for self-host).
7. `scripts/cloud-deploy.sh` one-liner installer.
8. Docs: `docs/cloud/self-host.md` step-by-step.

**Explicitly OUT of scope for v2.6:** Stripe, multi-tenant DB, user accounts, rate limits, billing. Those are v3.0.

### 9.7 v3.0.0 "Branch" — CoW memory branching + SaaS multi-tenancy

**ETA:** Q3 2026 at earliest. Gated on:
- v2.6 adoption signal (≥500 self-host deployments)
- Sam's runway (needs pre-revenue or funding)
- Either Mays, Orbit Wars, or another cash injection

**What it does:**
1. **Memory branching** — git-like CoW over SQLite. Branch a memory state, diverge freely, merge or discard. "What if I forgot all my Python biases and approached this memory as a Rust expert" becomes a one-button operation.
2. **Multi-tenant SaaS** at `vestige.dev` / `app.vestige.dev`. Per-user DB shards, JWT auth + OAuth providers, Stripe subscriptions with entitlement gates, org membership, team shared memory with role-based access.

**Major subsystems required:**
- Storage layer rewrite for CoW semantics (or adopt Dolt/sqlcipher with branching).
- Auth: JWT + OAuth (Google, GitHub, Apple) + bcrypt fallback.
- Billing: Stripe subscriptions + webhooks + dunning.
- Admin dashboard: support, usage analytics, churn.
- Multi-region: at minimum US-east + EU (GDPR).
- Observability: Prometheus + Grafana + Sentry + Honeycomb tracing.

**Explicitly NOT a v2.x goal.** Any earlier attempt burns runway.

### 9.8 Summary roadmap table

| Version | Codename | Theme | Effort | Load-bearing for | ETA |
|---|---|---|---|---|---|
| v2.1 | Decide | Qwen3 embeddings | ~1 week | Retrieval quality + differentiation vs. Nomic | Days |
| **v2.2** | **Pulse** | **Subconscious cross-pollination** | **~1 week (mostly latent)** | **★ Viral launch moment** | **~2 weeks** |
| v2.3 | Rewind | Time machine (slider + pin) | ~2 weeks | Technical moat, impressive demo | ~5 weeks |
| v2.4 | Empathy | Frustration detection → arousal boost | ~1 week | **First Pro-tier gate candidate** | ~7 weeks |
| v2.5 | Grip | Cluster gestures | ~1 week | Polish | ~9 weeks |
| v2.6 | Remote | vestige-cloud self-host (5→24 tools + Streamable HTTP + Docker) | ~3 weeks | Foundation for SaaS; secondary Pro-tier gate | ~12 weeks |
| v3.0 | Branch | CoW branching + multi-tenant SaaS | ~3 months | Revenue | Q3 2026 at earliest |

---

## 10. Composition Map

For each v2.x feature, what existing primitives does it compose?

| Feature | Existing primitive | How composed |
|---|---|---|
| v2.2 Pulse | `dream()` + `ConsolidationScheduler` + `ConnectionDiscovered` event + Three.js `events.ts::mapEventToEffects` | Consume the already-firing events; add toast UI + richer synthesis payload |
| v2.3 Rewind slider | `state_transitions` append log + FSRS decay formula + `temporal.ts::retentionAtDate()` + existing `TimeSlider.svelte` stub | Retroactive state reconstruction + slider upgrade |
| v2.3 Rewind pins | `smart_ingest` patterns + new `pins` table | Thin new table + two new tools |
| v2.4 Empathy | `ArousalSignal` (already in ImportanceSignals 4-channel model) + middleware pattern + `ImportanceTracker` | New middleware layer feeds existing arousal channel |
| v2.5 Grip | `find_duplicates` clustering + `promote`/`demote` + v2.0.8 Three.js node picking | Cluster-level wrapper over per-node operations |
| v2.6 Remote | v2.0.7 MCP tool implementations + vestige-cloud Feb skeleton + Axum | Port tools; implement Streamable HTTP; containerize |
| v3.0 Branch | Requires new CoW storage layer — **no existing primitive composes here** | Greenfield storage rewrite |
| v3.0 SaaS | Requires new auth + billing + multi-tenancy — **no existing primitive composes** | Greenfield |

**Key insight:** v2.2-v2.6 are all ≥60% latent in existing primitives. v3.0 is the first release that requires significant greenfield work. This is why sequencing matters: ride the existing primitives to revenue, then greenfield.

---

## 11. Risks & Known Gaps

### 11.1 Technical

| Risk | Impact | Mitigation |
|---|---|---|
| `ort-sys 2.0.0-rc.11` prebuilt gaps (Intel Mac dropped, Windows MSVC with usearch 2.24 broken) | Fewer platforms ship | Wait for ort-sys 2.1; or migrate to Candle throughout (v2.1 Qwen3 already uses Candle) |
| `usearch` pinned to 2.23.0 (2.24 regression on MSVC) | Windows build fragility | Monitor usearch#746 |
| fastembed model download (~130MB for Nomic, ~500MB for Qwen3) on first run blocks sandboxed Xcode | UX friction | Cache at `~/Library/Caches/com.vestige.core/fastembed` — documented in Xcode guide; pre-download from terminal once |
| Tool count drift (23 vs 24 across docs) | User trust | Reconciled in v2.0.7 (`docs: tool-count reconciliation`) |
| Large build times (cargo release 2-3 min incremental, 6+ min clean) | Slow iteration | M3 Max arriving Apr 20 will halve this |
| `include_dir!` bakes dashboard build into binary at compile time | Have to rebuild Rust to update dashboard | Accept as design; HMR via `pnpm dev` for iteration |

### 11.2 Product

| Risk | Impact | Mitigation |
|---|---|---|
| OSS-growth-before-revenue means months of zero cash | Sam can't pay rent | Mays May 1 ($400K+), Orbit Wars June 23 ($5K × top 10), part-time Wrigley Field during Cubs season |
| `deep_reference` is the crown jewel but rarely invoked | Users don't discover it | `CLAUDE.md` flags it; v2.2 Pulse farms the viral moment to drive awareness |
| Subconscious Pulse may fire too often or too rarely | User annoyance or missed value | Rate limit: max 1 pulse per 15 min; user-adjustable in settings |
| Emotional tagging may over-fire (every caps lock = frustration?) | False positives | Require ≥2 signals (retry + caps, or retry + correction) before boost |
| v3.0 SaaS burns runway if started too early | Business-ending | Gated on v2.6 adoption + cash injection |
| Copycat risk (Zep, Cognee, etc.) cloning Vestige's features | Eroded differentiation | AGPL-3.0 protects network use; neuroscience depth is hard to fake; time slider + subconscious pulse are visible moats |
| Cross-IDE MCP standard changes (Streamable HTTP spec moved 2024-11-05 → 2025-06-18) | Breaking transport changes | v2.6 implements the newer spec; keep 2024-11-05 as backward-compat alias |

### 11.3 Known UI gaps (`docs/launch/UI_ROADMAP_v2.1_v2.2.md`)

- **26% of MCP tools have zero UI surface** (e.g., `codebase`, `find_duplicates`, `backup`, `export`, `gc`, `restore` — all power-user only).
- **28% of cognitive modules have no visualization** (SynapticTagging, HippocampalIndex, ContextMatcher, CrossProjectLearner, etc.).
- The rainbow-bursted Rac1 cascade in the graph has no numeric "how many neighbours did it touch" display.
- `intention` shows but doesn't let you edit/snooze from the UI.
- `deep_reference` is unreachable from the dashboard (it only surfaces via MCP tool calls).

---

## 12. Viral / Launch / Content Plan

### 12.1 Content cadence (fixed)

**Mon–Fri till June 13 graduation:**
- 1-2 posts/day across Twitter + LinkedIn + Hacker News + Reddit r/LocalLLaMA + r/selfhosted
- Weekly YouTube long-form (Friday release)

### 12.2 Per-release launch playbook

For every v2.x release:
1. **Monday:** Tag + release + content drop (tweet with 30-90s demo video + HN post).
2. **Tuesday:** LinkedIn long-form + Reddit cross-post.
3. **Wednesday:** Follow-up tweet thread (deep-dive on one specific feature).
4. **Thursday:** Engage with feedback; close issues; publish patch if needed.
5. **Friday:** YouTube long-form (15-25 min walkthrough). Next week's release work continues.

### 12.3 Viral load-bearing moments

- **v2.2 "Pulse" launch:** The single biggest viral bet. Subconscious cross-pollination demo → HN front page → Twitter thread → YouTube 10-min walkthrough.
- **v2.3 "Rewind" time slider:** Highly tweet-friendly. Screen recording of sliding back through memory decay.
- **Jarrett Ye (FSRS creator, user L-M-Sherlock) outreach:** Already a stargazer. Email him Sunday night (US time) = Monday AM Beijing with the v2.2 Pulse demo. If he retweets → FSRS community (Anki, maimemo) amplifies.

### 12.4 Issue #36 (hooks-for-automatic-memory)

Outstanding from desaiuditd. Response plan:
1. Thank him publicly in the issue.
2. Acknowledge the feature as valid and scoped for v2.2/v2.3.
3. Open a linked sub-issue: "v2.2: Auto-memory hooks" tied to Pulse work.

### 12.5 Monetization gates

**Two candidate first-gates:**
1. **v2.4 Empathy (Emotional tagging)** — invisible to OSS demos, strong retention, clean paid-feature framing ("Vestige notices when you're frustrated; free tier gets 100 detection events/month, Pro gets unlimited + frustration hotspot analytics").
2. **v2.6 Remote (Cloud self-host binary)** — "free binary forever; paid-tier cloud-managed deploy with backups + observability + multi-device sync."

Pick after v2.2 viral signal tells us whether retention or convenience is the weaker link.

---

## 13. How AI Agents Should Consume This Doc

### 13.1 First-time read protocol

If this is the first time you're seeing Vestige:
1. Read Part 0 (Executive Summary) + Part 1 (What Vestige Is). That's 3 minutes.
2. Read Part 9 (The Plan). That's 10 minutes.
3. Bookmark Parts 3-6 for reference.

### 13.2 When Sam asks you to plan a feature

1. Check Part 9 — is it already scoped? If yes, that section IS your spec.
2. If not, work it into the existing roadmap: which version should it ship in, what primitives does it compose (Part 10), what risks apply (Part 11)?
3. Follow the `/Users/entity002/.claude/rules/active-synthesis.md` protocol (6 mandatory behaviors): compose, don't summarize.

### 13.3 When Sam asks you to implement

1. Find the exact file paths in Parts 3-5.
2. Check existing test coverage (Part 5.9 for dashboard, §3.11 for core).
3. Before claiming something exists, grep or read the source — memory alone is insufficient (per `CLAUDE.md` SCOUR rule).
4. Rust 1.95 toolchain — be aware of the new lints (`unnecessary_sort_by`, `collapsible_match`).

### 13.4 When Sam asks for strategic advice

- Apply the `/Users/entity002/.claude/rules/cross-reference.md` rule: check evidence from the exact setup before recommending.
- Apply the `always-positive-energy` rule: recommend the BEST path, not the safest.
- This doc's Part 9 is the committed roadmap. Deviate only with explicit justification.

### 13.5 Load-bearing files to never forget

- `/Users/entity002/Developer/vestige/CLAUDE.md` — project-level Claude instructions.
- `/Users/entity002/.claude/rules/active-synthesis.md` — 6 mandatory synthesis behaviors.
- `/Users/entity002/.claude/rules/cross-reference.md` — exact-setup evidence rule.
- `/Users/entity002/CLAUDE.md` — global Claude instructions (SCOUR + always-positive-energy).
- `/Users/entity002/Developer/vestige/docs/launch/UI_ROADMAP_v2.1_v2.2.md` — prior UI research compilation.
- **This file** — `/Users/entity002/Developer/vestige/docs/VESTIGE_STATE_AND_PLAN.md`.

---

## 14. Glossary & Citations

### 14.1 Acronyms

| Term | Meaning |
|---|---|
| **MCP** | Model Context Protocol — JSON-RPC protocol for AI tool integration (Anthropic, 2024) |
| **FSRS** | Free Spaced Repetition Scheduler — algorithm by Jarrett Ye (maimemo), generation 6 |
| **PE Gating** | Prediction Error Gating — decide CREATE/UPDATE/SUPERSEDE by similarity threshold |
| **SIF** | Suppression-Induced Forgetting — Anderson 2025 |
| **Rac1** | Rho-family GTPase — actin-destabilization mediator of cascade decay (Cervantes-Sandoval & Davis 2020) |
| **SWR** | Sharp-wave ripple — hippocampal replay pattern used by Vestige's dream cycle |
| **HNSW** | Hierarchical Navigable Small World — graph index for fast approximate nearest neighbour |
| **CoW** | Copy-on-write — storage technique for cheap branching |
| **AGPL** | Affero General Public License — copyleft including network use |

### 14.2 Neuroscience citations

- Anderson, M. C. (2025). Suppression-induced forgetting — top-down inhibitory control of retrieval.
- Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can cause forgetting.
- Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old theory of stimulus fluctuation. — dual-strength model.
- Brown, R., & Kulik, J. (1977). Flashbulb memories.
- Cervantes-Sandoval, I., & Davis, R. L. (2020). Rac1-mediated forgetting.
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing.
- Frey, U., & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation.
- Friston, K. J. (2010). The free-energy principle: a unified brain theory.
- Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval.
- Teyler, T. J., & Rudy, J. W. (2007). The hippocampal indexing theory.
- Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes.

### 14.3 Technical citations

- MCP Spec (2025-06-18 Streamable HTTP): https://modelcontextprotocol.io/specification
- FSRS-6: https://github.com/open-spaced-repetition/fsrs-rs
- Nomic Embed Text v1.5: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Qwen3 Embed: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- USearch: https://github.com/unum-cloud/usearch
- Jina Reranker v1 Turbo: https://huggingface.co/jinaai/jina-reranker-v1-turbo-en

---

**End of document.** Length-check: ~16,500 words / ~110 KB markdown. This is the single-page briefing that lets any AI agent plan the next phase of Vestige without having to re-read the repository.
