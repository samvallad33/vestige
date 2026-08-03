# Vestige v2.0 Launch — Show HN + Cross-Posts

---

## 1. Hacker News — Show HN

### Title (76 chars)

```
Show HN: Vestige – FSRS-6 spaced repetition as long-term memory for AI agents
```

### Body (first comment)

```
Hi HN,

I built Vestige because every AI conversation starts from zero. Your AI has no
memory of yesterday. I wanted to fix that with real science, not just a vector
database with a wrapper.

**What it is:** A memory server for AI agents (MCP protocol). It sits between
you and your AI — Claude, Cursor, VS Code Copilot, etc. — and gives it genuine
long-term memory with cognitive-science-backed forgetting, strengthening, and
retrieval. Written in Rust, 100% local, single 22MB binary.

**The neuroscience stack:**

- **FSRS-6 spaced repetition** — the same 21-parameter power-law forgetting
  model behind Anki's best algorithm, trained on 700M+ reviews. Memories decay
  along empirically-validated curves instead of living forever with equal weight.

- **Prediction Error Gating** — on ingest, new content is compared against all
  existing memories. If similarity >92%, it reinforces. 75-92%, it merges.
  <75%, it creates. This is inspired by how the brain decides what's worth
  encoding vs. what's redundant.

- **Dual-strength model** (Bjork & Bjork, 1992) — each memory tracks storage
  strength (how well-encoded, only increases) and retrieval strength (how
  accessible right now, decays over time). A memory can be well-stored but hard
  to retrieve, like a name on the tip of your tongue.

- **Testing Effect** — retrieval is audit-only; explicit positive feedback
  strengthens memories that proved useful. This applies a robust cognitive
  finding (Roediger & Karpicke, 2006) without pinning stale results.

- **Synaptic Tagging** (Frey & Morris, 1997) — when something important happens,
  it retroactively strengthens memories from the surrounding time window (default:
  9 hours back, 2 hours forward). This models how the brain consolidates memories
  during waking hours.

- **Spreading Activation** (Collins & Loftus, 1975) — searching for "React hooks"
  surfaces "useEffect" memories through semantic similarity, even without keyword
  overlap.

- **Memory Dreaming** — offline consolidation that replays recent memories to
  discover hidden connections and synthesize insights. Inspired by hippocampal
  replay during sleep.

**v2.0 adds:**

- 3D neural visualization dashboard (SvelteKit + Three.js) — watch memories
  pulse when accessed, burst particles on creation, golden flash lines when
  connections form. GPU instanced rendering handles 1000+ nodes at 60fps.

- WebSocket event bus — every cognitive operation (search, dream, consolidation,
  decay) broadcasts real-time events to the dashboard.

- HyDE query expansion — template-based Hypothetical Document Embeddings that
  classify query intent into 6 types, expand into 3-5 variants, and average
  the embedding vectors. Dramatically improves conceptual search.

- Everything compiles into a single 22MB binary. The SvelteKit dashboard is
  embedded via Rust's `include_dir!` macro. No Docker, no Node runtime, no
  external services.

**Numbers:** 77,840 lines of Rust, 734 tests, 29 cognitive modules, 21 MCP
tools, search under 50ms for 1000 memories (SQLite FTS5 + USearch HNSW).

**What it is NOT:** This is not RAG. RAG treats memory as a static database —
chunk everything, embed it, top-k retrieve. Vestige treats memory as a dynamic
cognitive system. Memories decay. Using them makes them stronger. Important
events retroactively strengthen recent memories. Irrelevant memories fade. The
system evolves.

The embedding model (Nomic Embed Text v1.5) runs locally via ONNX. After the
first-run model download (~130MB), there are zero network requests. No
telemetry, no analytics, no phoning home.

I've been using this daily for 2 months and the experience is genuinely different.
Claude remembers my coding patterns, my architectural decisions, my preferences.
New sessions start with context instead of a blank slate.

Source: https://github.com/samvallad33/vestige

Happy to answer any questions about the cognitive science, the Rust architecture,
or MCP in general.
```

---

## 2. Prepared FAQ for HN Comments

### Q: "How is this different from just shoving everything into a vector database?"

```
The core difference: a vector database gives everything equal weight forever.
Vestige applies forgetting.

In a vector DB, a note from 6 months ago you never referenced sits alongside
critical context from yesterday, both equally retrievable. Over time your
retrieval quality degrades because the signal-to-noise ratio gets worse.

Vestige uses FSRS-6 (the same algorithm as Anki's best spaced repetition mode)
to model forgetting curves. Memories explicitly confirmed useful get stronger
(Testing Effect).
Memories you ignore fade (power-law decay). Important events retroactively
strengthen nearby memories (Synaptic Tagging).

The result is a system where retrieval quality improves over time instead of
degrading. The AI surfaces what's actually relevant, not just what's
semantically closest.

It also deduplicates on ingest (Prediction Error Gating) — if you try to store
something 92%+ similar to existing memory, it reinforces instead of creating a
duplicate. This keeps the knowledge base clean without manual maintenance.
```

### Q: "Why not just use Claude's built-in memory / ChatGPT memory?"

```
Two reasons: control and science.

Control: Native AI memory is a black box on someone else's servers. You can't
see what was stored, how it decays, or export it. Vestige stores everything in
a local SQLite database you own. You can query it directly, back it up, export
as JSON, or delete it entirely.

Science: Native memory implementations are proprietary. We have no idea what
algorithm they use for retention or retrieval. Vestige uses published research
— FSRS-6 (power-law forgetting, 21 parameters, trained on 700M Anki reviews),
dual-strength model (Bjork & Bjork 1992), encoding specificity (Tulving 1973).
These are well-studied, empirically-validated models.

They also work simultaneously — Claude's native memory handles general context,
Vestige handles structured knowledge with explicit cognitive science.
```

### Q: "77K lines of Rust seems like a lot for a memory system. What's in there?"

```
Fair question. Roughly:

- ~22K: fastembed (vendored fork of the embedding library, ONNX inference)
- ~15K: 29 cognitive modules (FSRS-6, prediction error gating, synaptic
  tagging, spreading activation, dreaming, hippocampal index, etc.)
- ~12K: MCP server + 21 tool implementations
- ~8K: Storage layer (SQLite, FTS5, HNSW vector index, migrations)
- ~7K: SvelteKit dashboard (TypeScript/Svelte, embedded in binary)
- ~6K: Tests (734 tests across core + mcp + e2e + doctests)
- ~5K: Search pipeline (hybrid BM25+semantic, RRF fusion, HyDE, reranker)
- ~3K: Dashboard backend (Axum, WebSocket, REST API, event system)

Is it over-engineered? Maybe. But each cognitive module implements a specific
finding from memory research. The complexity comes from faithfully modeling how
memory actually works, not from unnecessary abstraction.
```

### Q: "Does FSRS-6 actually make a difference, or is it just a gimmick?"

```
FSRS-6 is the state of the art in spaced repetition. It was developed by the
open-spaced-repetition group, trained on 700M+ Anki reviews, and benchmarks
30% more efficient than SM-2 (the algorithm most SRS apps use, which dates
to 1987).

The key insight is the forgetting model. SM-2 uses exponential decay, which
doesn't match empirical data. FSRS-6 uses a power-law curve:

  R(t, S) = (1 + factor * t / S)^(-w20)

Power-law forgetting has been consistently demonstrated in memory research
since Wixted & Ebbesen (1991). The difference matters practically — exponential
decay predicts memories fall off a cliff, while power-law decay predicts a long
tail where old memories can still be retrieved.

For an AI memory system, this means old but important memories don't vanish.
They fade slowly and can be revived with explicit positive feedback (the Testing Effect).
```

### Q: "Does this work with models other than Claude?"

```
Yes. Vestige speaks MCP (Model Context Protocol), which is supported by Claude,
Cursor, VS Code Copilot, JetBrains, Windsurf, and others. Any MCP-compatible
client can use it.

The CLAUDE.md configuration in the repo tells the AI when and how to use the
memory tools, but the underlying server is model-agnostic. You could write
equivalent instructions for any model that supports MCP tool calling.
```

### Q: "Why AGPL-3.0?"

```
To prevent cloud providers from hosting Vestige as a competing service without
contributing back. AGPL requires that if you serve the software over a network,
you must open-source your modifications.

The core memory system is fully open source and always will be. If you run it
locally (which is the intended use case), AGPL is functionally identical to GPL.
```

### Q: "What's the performance like? SQLite seems like it would be slow for vectors."

```
SQLite handles the keyword search (FTS5 — very fast). Vector search uses USearch
HNSW with int8 quantization, which is separate from SQLite.

Benchmarks on an M1 MacBook Pro:

  100 memories:    <10ms search
  1,000 memories:  <50ms search
  10,000 memories: <200ms search
  100,000 memories: <1s search

  cosine_similarity: 296ns
  RRF fusion: 17µs
  Embedding generation: ~100ms per memory (only on ingest)

For personal use (hundreds to a few thousand memories), search is essentially
instant. The bottleneck is embedding generation, which only happens when storing
new memories.
```

### Q: "How does the 3D dashboard work? Is it practical or just eye candy?"

```
Both, honestly. The 3D force-directed graph is genuinely useful for seeing
clusters of related knowledge, discovering memories you forgot about, and
watching the dreaming process in real-time (memories pulse purple as they're
replayed).

But the primary UX for interacting with memories is through your AI. The
dashboard is a monitoring and exploration tool, not the main interface. You
talk to Claude/Cursor/etc normally, and Vestige handles memory in the
background via MCP tool calls.

The dashboard is embedded in the binary via include_dir! — no separate server.
It's served on localhost:3927/dashboard alongside the MCP stdio server.
```

### Q: "Is the 'dreaming' feature actually doing anything useful?"

```
It does three things:

1. Replays recent memories and computes pairwise semantic similarity to
   discover connections you didn't explicitly create.

2. Identifies memories that should be linked based on content overlap,
   creating an associative network.

3. Synthesizes short insights from clusters of related memories.

Is it as sophisticated as hippocampal replay during NREM sleep? No. It's an
engineering approximation that runs the same kind of "offline processing" —
reviewing, connecting, consolidating — that biological memory consolidation
does during sleep. The connections it discovers are sometimes genuinely
surprising and useful.
```

---

## 3. Reddit Cross-Posts

### r/rust

**Title:** `Vestige v2.0 — 77K LOC Rust memory system with FSRS-6, HNSW, Axum WebSockets, and an embedded SvelteKit dashboard in a 22MB binary`

**Body:**

```markdown
I've been building Vestige for the past few months and just shipped v2.0. It's
a cognitive memory system for AI agents that implements neuroscience-backed
memory algorithms in pure Rust.

**Rust-specific highlights:**

- **Single binary deployment**: SvelteKit dashboard compiled to static files,
  then embedded into the Rust binary via `include_dir!`. The entire system —
  MCP server, HTTP dashboard, WebSocket event bus, embedding inference — ships
  as one 22MB binary.

- **fastembed vendored fork**: We vendor a fork of the fastembed-rs crate for
  ONNX embedding inference (Nomic Embed Text v1.5, 768-dim vectors truncated
  to 256 via Matryoshka). Feature flags for Nomic v2 MoE and Metal GPU
  acceleration.

- **Axum 0.8 + tokio broadcast**: Dashboard runs on Axum with WebSocket
  upgrade at `/ws`. A single `tokio::broadcast::channel(1024)` propagates
  events from the MCP stdio server to all connected dashboard clients.

- **USearch HNSW**: Vector similarity search via USearch with int8
  quantization (M=16, efConstruction=128, efSearch=64). Criterion benchmarks:
  cosine_similarity at 296ns, RRF fusion at 17µs.

- **SQLite + FTS5**: rusqlite 0.38 with WAL mode, reader/writer connection
  split, FTS5 porter tokenizer for keyword search. Interior mutability via
  `Mutex<Connection>` — all Storage methods take `&self`.

- **Rust 2024 edition**: Using `use<'_>` captures in RPITIT and the latest
  edition features. MSRV 1.85.

- **Release profile**: `lto = true`, `codegen-units = 1`, `opt-level = "z"`,
  `strip = true` gets the binary down to 22MB including embedded assets.

- **734 tests**: 352 core + 378 mcp + 4 doctests. Zero warnings.

**Architecture:**

```
MCP Client (Claude/Cursor/etc)
    |  stdio JSON-RPC
    v
McpServer (rmcp 0.14)
    |
    +---> Arc<Storage> (SQLite + HNSW)
    |
    +---> Arc<Mutex<CognitiveEngine>> (29 modules)
    |
    +---> broadcast::Sender<VestigeEvent>
              |
              v
          Axum Dashboard (port 3927)
              |
              +---> /ws (WebSocket)
              +---> /api/* (REST)
              +---> /dashboard/* (SvelteKit static)
```

The cognitive engine implements FSRS-6 spaced repetition, prediction error
gating, synaptic tagging, spreading activation, and memory dreaming. Each
module is a stateful struct initialized once at startup and shared via Arc.

**What I'd do differently:** The fastembed vendoring is the ugliest part of the
codebase. ONNX Runtime bindings are notoriously painful in Rust, and I spent
more time fighting `ort` than any other dependency. If I started over, I might
explore `candle` as the primary backend instead of ORT.

Source: https://github.com/samvallad33/vestige
License: AGPL-3.0

Happy to discuss any of the Rust architecture decisions.
```

---

### r/ClaudeAI

**Title:** `Vestige v2.0 "Cognitive Leap" — give Claude real long-term memory with neuroscience-backed forgetting, a 3D dashboard, and 21 MCP tools`

**Body:**

```markdown
Vestige gives Claude persistent memory across sessions using real cognitive
science instead of just dumping everything into a database.

**The problem it solves:** Every Claude conversation starts from zero. Even with
native memory, you have no control over what's stored, how it decays, or where
it lives. Vestige gives Claude a proper long-term memory system that runs 100%
locally on your machine.

**What makes it different from other MCP memory servers:**

- **Memories decay like real memories.** FSRS-6 spaced repetition (the same
  algorithm Anki uses) models forgetting curves. Memories you use get stronger.
  Memories you ignore fade. Important events retroactively strengthen recent
  memories.

- **Smart deduplication.** When Claude tries to save something similar to what
  it already knows, Prediction Error Gating decides whether to create, merge,
  or just reinforce the existing memory. No manual cleanup needed.

- **29 cognitive modules** implementing findings from memory research: dual-
  strength model, testing effect, synaptic tagging, spreading activation,
  context-dependent retrieval, memory dreaming.

**v2.0 new features:**

- **3D Memory Dashboard** at localhost:3927/dashboard — watch Claude's mind in
  real-time. Memories pulse when accessed, burst particles on creation, golden
  lines when connections form. SvelteKit + Three.js with bloom post-processing.

- **Real-time event bus** — every cognitive operation (search, dream,
  consolidation) broadcasts WebSocket events to the dashboard.

- **HyDE query expansion** — dramatically better search for conceptual queries.

- **Single 22MB binary** — everything embedded, no Docker, no Node, no cloud.

**Setup (2 minutes):**

```bash
npm install -g vestige-mcp-server
claude mcp add vestige vestige-mcp -s user
```

Then add the CLAUDE.md instructions from the repo to tell Claude how to use
memory tools automatically.

**What it's like in practice:** After 2 months of daily use, Claude remembers
my coding patterns, my architectural decisions, my preferences across every
project. New sessions start with context instead of a blank slate. It knows I
prefer Rust over Go, that I use Tailwind, and that my last debugging session
on Project X ended with a tricky race condition in the WebSocket handler.

It's the difference between talking to someone with amnesia vs. someone who
actually knows you.

21 MCP tools. 77,840 lines of Rust. 734 tests. Works with Claude Code, Claude
Desktop, Cursor, VS Code Copilot, JetBrains, Windsurf, and Xcode.

Source: https://github.com/samvallad33/vestige

Happy to answer questions or help with setup.
```

---

### r/LocalLLaMA

**Title:** `Vestige v2.0 — local-first AI memory server with FSRS-6 spaced repetition, ONNX embeddings, and zero cloud dependency (77K LOC Rust, 22MB binary)`

**Body:**

```markdown
Vestige is a memory system for AI agents that runs entirely on your machine.
No cloud, no API keys, no telemetry. After the first-run embedding model
download (~130MB), it makes zero network requests.

**Why this matters for local LLM setups:**

Most memory/RAG solutions assume cloud embeddings (OpenAI, Cohere, etc.).
Vestige embeds locally via ONNX (Nomic Embed Text v1.5, 768-dim vectors) and
stores everything in a local SQLite database. If you're already running local
models, your memory system should be local too.

**The cognitive science angle:**

This isn't just another vector database wrapper. It implements real memory
algorithms:

- **FSRS-6**: The state-of-the-art spaced repetition algorithm (power-law
  forgetting, 21 parameters, trained on 700M+ Anki reviews). Memories decay
  naturally instead of living forever.

- **Prediction Error Gating**: On ingest, compares new content against existing
  memories. Creates/merges/reinforces based on novelty. Prevents bloat.

- **Testing Effect**: Explicit positive feedback strengthens a memory after it
  proves useful. Retrieval alone never pins stale information.

- **Dual-strength model**: Storage strength (how well-encoded, never decreases)
  vs. retrieval strength (how accessible now, decays over time). Mimics the
  tip-of-tongue phenomenon.

- **Memory Dreaming**: Offline consolidation that replays memories to discover
  connections. Inspired by hippocampal replay during sleep.

**Technical stack:**

- Rust, single 22MB binary
- Nomic Embed Text v1.5 via ONNX (local, ~130MB model)
- Optional: Nomic v2 MoE (475M params, feature flag), Metal GPU (Apple Silicon)
- SQLite FTS5 for keyword search + USearch HNSW for vector search
- MCP protocol (works with Claude, Cursor, VS Code Copilot, etc.)
- Axum HTTP + WebSocket for dashboard
- SvelteKit + Three.js 3D visualization (embedded in binary)

**v2.0 highlights:**

- 3D force-directed memory graph with real-time WebSocket events
- HyDE query expansion (template-based hypothetical document embeddings)
- FSRS decay visualization with retention curves
- 734 tests, 29 cognitive modules, 21 tools
- fastembed 5.11 with feature flags for Nomic v2 MoE + Qwen3 reranker

**Performance:**

- Search: <50ms for 1K memories, <200ms for 10K
- Embedding: ~100ms per memory (ingest only)
- cosine_similarity: 296ns (Criterion benchmark)
- Memory: ~100MB for 1K memories, ~300MB for 10K

**For local model users specifically:** Vestige speaks MCP (Model Context
Protocol). If your local model setup supports MCP tool calling, it can use
Vestige directly. The CLAUDE.md instructions in the repo tell the model when
and how to use memory tools — you can adapt these for any model.

The embedding model downloads from Hugging Face on first run and caches
locally. After that, fully air-gapped.

Source: https://github.com/samvallad33/vestige
License: AGPL-3.0 (use freely for local/personal use, cloud service requires
source disclosure)

This is a solo project — feedback, issues, and contributions are very welcome.
```

---

## 4. Posting Strategy

### Timing

- **Show HN**: Tuesday or Wednesday, 8-10 AM EST (peak engagement window)
- **r/rust**: Same day, 1-2 hours after HN post goes up
- **r/ClaudeAI**: Same day, stagger by 1 hour
- **r/LocalLLaMA**: Same day or next morning

### Rules to Follow

- **HN**: Title is the post. First comment is the body above. Respond to every
  comment within 30 minutes for the first 3 hours. Be humble, technical,
  transparent about limitations. Never say "AI-powered" or "game-changer."
- **Reddit**: Each subreddit gets a tailored post emphasizing what that
  community cares about. No cross-linking between posts. Engage authentically.
- **General**: Lead with the science, not the product. Let the tech speak.
  Acknowledge competitors honestly. Never disparage alternatives.

### Key Messaging Points

1. This is a solo project built on published research, not a startup pitch
2. The neuroscience is real but honestly described (some modules are faithful
   implementations, some are engineering heuristics inspired by research)
3. 100% local, zero cloud — this is a feature, not a limitation
4. The 3D dashboard is a genuine exploration tool, not just eye candy
5. FSRS-6 is the differentiator — no other AI memory system uses real spaced
   repetition

### What NOT to Say

- "Revolutionary" / "game-changing" / "paradigm shift"
- "AI-powered" (it IS AI infrastructure, don't label it that way)
- Anything negative about Mem0, Cognee, or other competitors
- Claims about being "the best" at anything
- Marketing language of any kind
