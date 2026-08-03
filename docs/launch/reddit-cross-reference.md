# Reddit Launch Posts — cross_reference Tool

## Post 1: r/ClaudeAI (Primary)

**Title:** `I built a tool that catches when Claude's memories contradict each other before it answers. It's been wrong 0 times since.`

---

I've been building Vestige — an MCP memory server that gives Claude persistent memory across sessions. FSRS-6 spaced repetition (the Anki algorithm), 29 neuroscience-inspired modules, single Rust binary. 1,111 memories stored. It works.

But last week it almost cost me hours of debugging.

Claude confidently told me a benchmark notebook should use `--enable-prefix-caching` with vLLM. I trusted it. The notebook crashed. Burned a daily submission.

The problem? I had TWO memories:
- **January**: "prefix caching crashes with our vLLM build"  
- **March**: "prefix caching works with the newer vLLM build"

Claude found both. Picked the wrong one. Gave me a confident wrong answer based on the January memory. The March memory was correct — but Claude had no way to know they conflicted.

So I built `cross_reference`.

Now before answering any factual question, Claude calls:

```
cross_reference({ query: "should I use --enable-prefix-caching with vLLM?" })
```

And gets back:

```json
{
  "status": "contradictions_found",
  "confidence": 0.35,
  "guidance": "WARNING: 1 contradiction detected across 12 memories. 
               The newer memory is likely more accurate.",
  "contradictions": [
    {
      "newer": {
        "date": "2026-03-18",
        "preview": "Switched to a newer vLLM build that supports --enable-prefix-caching..."
      },
      "older": {
        "date": "2026-01-15", 
        "preview": "prefix caching crashed with our older local vLLM build..."
      },
      "recommendation": "Trust the newer memory. Consider demoting the older one."
    }
  ],
  "timeline": [ /* 12 memories sorted newest-first */ ]
}
```

Claude sees the contradiction BEFORE answering. Trusts the March memory. Gets it right.

**I've been using it for 2 days. It's caught contradictions I didn't even know I had.** Old project decisions that got reversed. Config values that changed. Library versions that got upgraded. All sitting in memory, waiting to give me wrong answers.

### How it works under the hood:

1. **Broad retrieval** — pulls up to 50 memories related to the query
2. **Cross-encoder reranking** — filters for quality (Jina Reranker v1 Turbo)
3. **Pairwise contradiction detection** — every memory pair checked for negation patterns ("don't" vs "do", "deprecated" vs "recommended") and correction signals ("now uses", "switched to", "replaced by")
4. **Topic gating** — only compares memories that share significant words (prevents false positives between unrelated memories)
5. **Confidence scoring** — agreements boost confidence, contradictions tank it, recency helps
6. **Timeline** — everything sorted newest-first so Claude sees the evolution
7. **Superseded list** — explicitly identifies which old memories should be demoted

### The bigger picture:

Context windows are now 1M tokens (Claude Opus 4.6, GPT-5.4). But bigger context doesn't fix this problem. The "Lost in the Middle" research shows accuracy drops from 92% to 78% at 1M tokens. More context = more chances for contradictions to slip through.

Memory systems need to be SMARTER, not just bigger. That's what Vestige does — 29 cognitive modules implementing real neuroscience:

- **FSRS-6** — memories naturally decay when unused, strengthen after explicit positive feedback (21 parameters trained on 700M+ reviews)
- **Prediction Error Gating** — only stores what's genuinely new (no duplicates)
- **Dream consolidation** — replays memories to discover hidden connections (yes, like sleep)
- **Spreading activation** — search for "auth bug" and find the related JWT update from last week
- **cross_reference** — the new tool that catches contradictions before they become wrong answers

### Stats:
- 22 MCP tools
- 746 tests, 0 failures
- Zero `unsafe` code
- Clean security audit (0 findings — AgentAudit verified)
- Single 22MB Rust binary — no Docker, no PostgreSQL, no cloud
- Works with Claude Code, Cursor, VS Code, Xcode 26.3, JetBrains, Windsurf

### Install (30 seconds):
```bash
# macOS Apple Silicon
npm install -g vestige-mcp-server
sudo mv vestige-mcp /usr/local/bin/
claude mcp add vestige vestige-mcp -s user
```

Then add to your CLAUDE.md:
```
Before answering factual questions, call cross_reference({ query: "the topic" })
to verify your memories don't contradict each other.
```

That's it. Your AI now fact-checks itself.

**GitHub:** https://github.com/samvallad33/vestige

I searched every memory MCP server out there — Mem0 (47K stars), Hindsight, Zep, Letta, OMEGA, Solitaire, MuninnDB. Some detect contradictions at ingest time (when you save). Nobody else gives the AI an on-demand tool to verify its own memories before answering, with confidence scoring and guidance on which memory to trust.

Happy to answer questions about the neuroscience, the architecture, or how to set it up.

---

## Post 2: r/LocalLLaMA (Cross-post, 2h later)

**Title:** `Your AI has 1000 memories. Some contradict each other. It doesn't know. I built the fix — 100% local, single Rust binary, zero cloud.`

---

The problem nobody talks about with AI memory:

You use a memory system. Over months, you accumulate 1000+ memories. Your project config changed 3 times. A library got deprecated. A decision got reversed. All those old memories are still there.

When your AI searches memory, it finds 5 results. Two of them disagree. The AI picks one — maybe the wrong one — and gives you a confident answer based on outdated information.

I built `cross_reference` to fix this. It's tool #22 in Vestige, my cognitive memory MCP server.

```
cross_reference({ query: "what database does the project use?" })
```

Returns:
```json
{
  "status": "contradictions_found",
  "confidence": 0.35,
  "contradictions": [{
    "newer": { "date": "2026-03", "preview": "Migrated to PostgreSQL..." },
    "older": { "date": "2026-01", "preview": "Using SQLite for the backend..." }
  }],
  "guidance": "WARNING: Trust the newer memory."
}
```

The AI sees the conflict. Picks the right one. Every time.

**What makes this different from RAG/vector search:**

| | RAG | Vestige |
|---|---|---|
| Storage | Store everything | Prediction Error Gating — only stores what's new |
| Retrieval | Nearest neighbor | 7-stage cognitive pipeline with reranking |
| Contradictions | Returns both, hopes for the best | **Detects and flags before answering** |
| Decay | Nothing expires | FSRS-6 — memories fade naturally |
| Duplicates | Manual dedup | Self-healing via semantic comparison |

**The stack:**
- Rust 2024 edition. Single 22MB binary. No Python, no Docker, no cloud.
- FSRS-6 spaced repetition (21 parameters, 700M+ reviews)
- 29 cognitive modules (spreading activation, synaptic tagging, dream consolidation, prediction error gating)
- 3D neural dashboard at localhost:3927 (Three.js, real-time WebSocket)
- MCP server — works with any AI that speaks MCP

**100% local. Your data never leaves your machine.**

```bash
npm install -g vestige-mcp-server
sudo mv vestige-mcp /usr/local/bin/
claude mcp add vestige vestige-mcp -s user
```

746 tests. Zero unsafe code. Clean security audit. AGPL-3.0.

GitHub: https://github.com/samvallad33/vestige

---

## Post 3: r/rust (Optional, technical audience)

**Title:** `I built a 22MB Rust binary that gives AI agents a brain — FSRS-6, 29 cognitive modules, 3D dashboard, and a new contradiction detection tool. 746 tests, zero unsafe.`

---

Vestige is a cognitive memory engine for AI agents. MCP server (stdio JSON-RPC + HTTP transport), SQLite WAL backend, USearch HNSW vector search, Nomic Embed v1.5 via fastembed (local ONNX, no API).

The latest addition: `cross_reference` — pairwise contradiction detection across memories at retrieval time. The AI calls it before answering factual questions to verify its memories don't disagree.

**Why Rust:**
- Single static binary (22MB with embedded SvelteKit dashboard via `include_dir!`)
- No runtime, no GC pauses during real-time search
- `tokio::sync::Mutex` for the cognitive engine, `std::sync::Mutex` for SQLite reader/writer split
- Zero `unsafe` blocks in the entire codebase
- `cargo test` runs 746 tests in 11 seconds

**Architecture:**
```
Axum 0.8 (dashboard + HTTP transport)
  ↕ WebSocket event bus (tokio::broadcast, 1024 capacity)
MCP Server (stdio JSON-RPC)
  → 22 tools dispatched via match on tool name
  → Arc<Storage> + Arc<Mutex<CognitiveEngine>>
SQLite WAL + FTS5 + USearch HNSW
  → fastembed 5.11 (Nomic Embed v1.5, 768D → 256D Matryoshka)
  → Jina Reranker v1 Turbo (cross-encoder, 38M params)
```

**Key crates:** rusqlite 0.38, axum 0.8, tokio 1, fastembed 5.11, usearch 2, chrono 0.4, serde 1, uuid 1, include_dir 0.7

**What I learned building it:**
- `OnceLock<Result<Mutex<TextEmbedding>, String>>` for lazy model initialization — the embedding model downloads ~130MB on first run, `OnceLock` ensures it only happens once and caches the error if it fails
- `floor_char_boundary()` saved me from a UTF-8 panic (content truncation with multi-byte chars)
- SQLite `PRAGMA journal_mode = WAL` + `synchronous = NORMAL` + `mmap_size = 268435456` gives surprisingly good concurrent read performance
- `try_lock()` pattern for cognitive features in search — if the lock is held (by dream consolidation), search falls back to simpler scoring instead of blocking

Clean security audit. Parameterized SQL everywhere. CSP headers on the dashboard. Constant-time auth comparison (`subtle::ConstantTimeEq`). File permissions 0o600/0o700.

GitHub: https://github.com/samvallad33/vestige  
AGPL-3.0 | 746 tests | 79K+ LOC

---

## Posting Strategy

| Subreddit | When | Expected Engagement |
|-----------|------|-------------------|
| r/ClaudeAI | First — 12-14 UTC (US morning) | High — direct audience for MCP tools |
| r/LocalLLaMA | 2h after r/ClaudeAI | High — local-first angle resonates here |
| r/rust | Same day, evening UTC | Medium — technical deep dive audience |
| r/MachineLearning | Next day if first two do well | Lower but prestigious |

**Title formula that works on Reddit:** Personal story + specific problem + "nobody else has this"

**DO NOT do:** Product-name-first titles, marketing speak, "introducing X", "check out my project"

**DO:** Lead with the PAIN ("my AI gave me wrong answers"), show the FIX (the JSON output), then reveal the tool.
