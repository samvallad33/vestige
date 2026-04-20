# 🚫 READ FIRST EVERY PROMPT — NO HEDGING

Sam's April 20 2026 correction: *"you NEVER LISTEN TO YOUR RULES, WHY ARE YOU ALWAYS BREAKING THE HEDGING/PESSIMISTIC RULE."*

When Sam commits to an execution path (content cadence, shipping velocity, outreach, launch), COMPUTE THE FORECAST AS IF THAT COMMITMENT IS MET. Do not discount by execution probability. Sam has earned the benefit of the doubt across 139 commits and 1,292 passing tests in 3 months.

**Forbidden in any response**: "has to be true", "has to convert", "realistic forecast" used to shrink, "contingent on" Sam's own execution, "not guaranteed", "may or may not", "aspiration not forecast", "temper your expectations", floors framed as reality vs. ceilings framed as fantasy.

**Required format for forecasts**: `Target X. Floor Y. Ceiling Z. Math: [line items]. Risks: [separate section]. Let's go.`

Full rule: `/Users/entity002/CLAUDE.md` (top banner) + `/Users/entity002/.claude/projects/-Users-entity002/memory/feedback_always_positive_energy.md`. Stop-hook enforcement: `/Users/entity002/.claude/hooks/synthesis-stop-validator.sh`.

---

# Vestige v2.0.4 — Cognitive Memory & Reasoning System

Vestige is your long-term memory AND reasoning engine. 29 stateful cognitive modules implement real neuroscience: FSRS-6 spaced repetition, synaptic tagging, prediction error gating, hippocampal indexing, spreading activation, reconsolidation, and dual-strength memory theory. **Use it automatically. Use it aggressively.**

**NEW: `deep_reference` — call this for ALL factual questions.** It doesn't just retrieve — it REASONS across memories with FSRS-6 trust scoring, intent classification, contradiction analysis, and generates a pre-built reasoning chain. Read the `reasoning` field FIRST.

---

## Session Start Protocol

Every conversation, before responding to the user:

```
session_context({
  queries: ["user preferences", "[current project] context"],
  context: { codebase: "[project]", topics: ["[current topics]"] },
  token_budget: 2000
})
```

Then check `automationTriggers` from response:
- `needsDream` → call `dream` (consolidates memories, discovers hidden connections)
- `needsBackup` → call `backup`
- `needsGc` → call `gc(dry_run: true)` then review
- totalMemories > 700 → call `find_duplicates`

Say "Remembering..." then retrieve context before answering.

> **Fallback:** If `session_context` unavailable: `search` × 2 → `intention` check → `system_status` → `predict`.

---

## Complete Tool Reference (23 Tools)

### session_context — One-Call Initialization
```
session_context({
  queries: ["user preferences", "project context"],  // search queries
  context: { codebase: "project-name", topics: ["svelte", "rust"], file: "src/main.rs" },
  token_budget: 2000,        // 100-100000, controls response size
  include_status: true,       // system health
  include_intentions: true,   // triggered reminders
  include_predictions: true   // proactive memory predictions
})
```
Returns: markdown context + `automationTriggers` + `expandable` IDs for on-demand retrieval.

### smart_ingest — Save Anything
**Single mode** — auto-decides CREATE/UPDATE/SUPERSEDE via Prediction Error Gating:
```
smart_ingest({
  content: "What to remember",
  tags: ["tag1", "tag2"],
  node_type: "fact",          // fact|concept|event|person|place|note|pattern|decision
  source: "optional reference",
  forceCreate: false          // bypass dedup when needed
})
```
**Batch mode** — save up to 20 items in one call (session end, pre-compaction):
```
smart_ingest({
  items: [
    { content: "Item 1", tags: ["session-end"], node_type: "fact" },
    { content: "Item 2", tags: ["bug-fix"], node_type: "fact" }
  ]
})
```
Each item runs the full cognitive pipeline: importance scoring → intent detection → synaptic tagging → hippocampal indexing → PE gating → cross-project recording.

### search — 7-Stage Cognitive Search
```
search({
  query: "search query",
  limit: 10,                  // 1-100
  min_retention: 0.0,         // filter by retention strength
  min_similarity: 0.5,        // minimum cosine similarity
  detail_level: "summary",    // brief|summary|full
  context_topics: ["rust", "debugging"],  // boost topic-matching memories
  token_budget: 3000,         // 100-100000, truncate to fit
  retrieval_mode: "balanced"  // precise|balanced|exhaustive (v2.1)
})
```
Retrieval modes: `precise` (fast, no activation/competition), `balanced` (default 7-stage pipeline), `exhaustive` (5x overfetch, deep graph traversal, no competition suppression).

Pipeline: Overfetch → Rerank (cross-encoder) → Temporal boost → Accessibility filter (FSRS-6) → Context match (Tulving 1973) → Competition (Anderson 1994) → Spreading activation. **Every search strengthens the memories it finds (Testing Effect).**

### memory — Read, Edit, Delete, Promote, Demote
```
memory({ action: "get", id: "uuid" })           // full node with all FSRS state
memory({ action: "edit", id: "uuid", content: "updated text" })  // preserves FSRS state, regenerates embedding
memory({ action: "delete", id: "uuid" })
memory({ action: "promote", id: "uuid", reason: "was helpful" })  // +0.20 retrieval, +0.10 retention, 1.5x stability
memory({ action: "demote", id: "uuid", reason: "was wrong" })     // -0.30 retrieval, -0.15 retention, 0.5x stability
memory({ action: "state", id: "uuid" })          // Active/Dormant/Silent/Unavailable + accessibility score
memory({ action: "get_batch", ids: ["uuid1", "uuid2", "uuid3"] })  // retrieve up to 20 full memories at once (v2.1)
```
Promote/demote does NOT delete — it adjusts ranking. Demoted memories rank lower; alternatives surface instead.
`get_batch` is designed for batch retrieval of expandable overflow IDs from search/session_context.

### codebase — Code Patterns & Architectural Decisions
```
codebase({ action: "remember_pattern", name: "Pattern Name",
  description: "How it works and when to use it",
  files: ["src/file.rs"], codebase: "project-name" })

codebase({ action: "remember_decision", decision: "What was decided",
  rationale: "Why", alternatives: ["Option A", "Option B"],
  files: ["src/file.rs"], codebase: "project-name" })

codebase({ action: "get_context", codebase: "project-name", limit: 10 })
// Returns: patterns, decisions, cross-project insights
```

### intention — Prospective Memory (Reminders)
```
intention({ action: "set", description: "What to do",
  trigger: { type: "context", topic: "authentication" },  // fires when discussing auth
  priority: "high" })

intention({ action: "set", description: "Deploy by Friday",
  trigger: { type: "time", at: "2026-03-07T17:00:00Z" },
  deadline: "2026-03-07T17:00:00Z" })

intention({ action: "set", description: "Check test coverage",
  trigger: { type: "context", codebase: "vestige", file_pattern: "*.test.*" } })

intention({ action: "check", context: { codebase: "vestige", topics: ["testing"] } })
intention({ action: "update", id: "uuid", status: "complete" })
intention({ action: "list", filter_status: "active" })
```

### dream — Memory Consolidation
```
dream({ memory_count: 50 })
```
5-stage cycle: Replay → Cross-reference → Strengthen → Prune → Transfer. Uses Waking SWR tagging (70% tagged + 30% random for diversity). Discovers hidden connections, generates insights, persists new edges to the activation network.

### explore_connections — Graph Traversal
```
explore_connections({ action: "associations", from: "uuid", limit: 10 })
// Spreading activation from a memory — find related memories via graph traversal

explore_connections({ action: "chain", from: "uuid-A", to: "uuid-B" })
// Build reasoning path between two memories (A*-like pathfinding)

explore_connections({ action: "bridges", from: "uuid-A", to: "uuid-B" })
// Find connecting memories that bridge two concepts
```

### predict — Proactive Retrieval
```
predict({ context: { codebase: "vestige", current_file: "src/main.rs",
  current_topics: ["error handling", "rust"] } })
```
Returns: predictions with confidence, suggestions, speculative retrievals, top interests. Uses SpeculativeRetriever's learned patterns from access history.

### importance_score — Should I Save This?
```
importance_score({ content: "Content to evaluate",
  context_topics: ["debugging"], project: "vestige" })
```
4-channel model: novelty (0.25), arousal (0.30), reward (0.25), attention (0.20). Composite > 0.6 = save it.

### find_duplicates — Dedup Memory
```
find_duplicates({ similarity_threshold: 0.80, limit: 20, tags: ["bug-fix"] })
```
Cosine similarity clustering. Returns merge/review suggestions.

### memory_timeline — Chronological Browse
```
memory_timeline({ start: "2026-02-01", end: "2026-03-01",
  node_type: "decision", tags: ["vestige"], limit: 50, detail_level: "summary" })
```

### memory_changelog — Audit Trail
```
memory_changelog({ memory_id: "uuid", limit: 20 })       // per-memory history
memory_changelog({ start: "2026-03-01", limit: 20 })      // system-wide
```

### memory_health — Retention Dashboard
```
memory_health()
```
Returns: avg retention, distribution buckets (0-20%, 20-40%, etc.), trend (improving/declining/stable), recommendation.

### memory_graph — Visualization Export
```
memory_graph({ query: "search term", depth: 2, max_nodes: 50 })
memory_graph({ center_id: "uuid", depth: 3, max_nodes: 100 })
```
Returns nodes with force-directed positions + edges with weights.

### deep_reference — Cognitive Reasoning Engine (v2.0.4) ★ USE THIS FOR ALL FACTUAL QUESTIONS
```
deep_reference({ query: "What port does the dev server use?" })
deep_reference({ query: "Should I use prefix caching with vLLM?", depth: 30 })
```
**THE killer tool.** 8-stage cognitive reasoning pipeline:
1. Broad retrieval + cross-encoder reranking
2. Spreading activation expansion (finds connected memories search misses)
3. FSRS-6 trust scoring (retention × stability × reps ÷ lapses)
4. Intent classification (FactCheck / Timeline / RootCause / Comparison / Synthesis)
5. Temporal supersession (newer high-trust replaces older)
6. Trust-weighted contradiction analysis (only flags conflicts between strong memories)
7. Relation assessment (Supports / Contradicts / Supersedes / Irrelevant per pair)
8. **Template reasoning chain** — pre-built natural language reasoning the AI validates

Parameters: `query` (required), `depth` (5-50, default 20).

Returns: `intent`, `reasoning` (THE KEY FIELD — read this first), `recommended` (highest-trust answer), `evidence` (trust-sorted), `contradictions`, `superseded`, `evolution`, `related_insights`, `confidence`.

`cross_reference` is a backward-compatible alias that calls `deep_reference`.

### Maintenance Tools
```
system_status()                              // health + stats + warnings + recommendations
consolidate()                                // FSRS-6 decay cycle + embedding generation
backup()                                     // SQLite backup → ~/.vestige/backups/
export({ format: "json", tags: ["bug-fix"], since: "2026-01-01" })
gc({ min_retention: 0.1, dry_run: true })    // garbage collect (dry_run first!)
restore({ path: "/path/to/backup.json" })
```

---

## Mandatory Save Gates

**You MUST NOT proceed past a save gate without executing the save.**

| Gate | Trigger | Action |
|------|---------|--------|
| **BUG_FIX** | After any error is resolved | `smart_ingest({ content: "BUG FIX: [error]\nRoot cause: [why]\nSolution: [fix]\nFiles: [paths]", tags: ["bug-fix", "project"], node_type: "fact" })` |
| **DECISION** | After any architectural/design choice | `codebase({ action: "remember_decision", decision, rationale, alternatives, files, codebase })` |
| **CODE_CHANGE** | After >20 lines or new pattern | `codebase({ action: "remember_pattern", name, description, files, codebase })` |
| **SESSION_END** | Before stopping or compaction | `smart_ingest({ items: [{ content: "SESSION: [summary]", tags: ["session-end"] }] })` |

---

## Trigger Words — Auto-Save

| User Says | Action |
|-----------|--------|
| "Remember this" / "Don't forget" | `smart_ingest` immediately |
| "I always..." / "I never..." / "I prefer..." | Save as preference |
| "This is important" | `smart_ingest` + `memory(action="promote")` |
| "Remind me..." / "Next time..." | `intention({ action: "set" })` |

---

## Cognitive Architecture

### Search Pipeline (7 stages)
1. **Overfetch** — 3x results from hybrid search (0.3 BM25 + 0.7 semantic, nomic-embed-text-v1.5 768D)
2. **Rerank** — Cross-encoder rescoring (Jina Reranker v1 Turbo, 38M params)
3. **Temporal** — Recency + validity window boosting (85% relevance + 15% temporal)
4. **Accessibility** — FSRS-6 retention filter (Active ≥0.7, Dormant ≥0.4, Silent ≥0.1)
5. **Context** — Tulving 1973 encoding specificity (topic overlap → +30% boost)
6. **Competition** — Anderson 1994 retrieval-induced forgetting (winners strengthen, competitors weaken)
7. **Activation** — Spreading activation side effects + predictive model + reconsolidation marking

### Ingest Pipeline
**Pre:** 4-channel importance scoring (novelty/arousal/reward/attention) + intent detection → auto-tag
**Store:** Prediction Error Gating: similarity >0.92 → UPDATE, 0.75-0.92 → UPDATE/SUPERSEDE, <0.75 → CREATE
**Post:** Synaptic tagging (Frey & Morris 1997, 9h backward + 2h forward) + hippocampal indexing + cross-project recording

### FSRS-6 (State-of-the-Art Spaced Repetition)
- Retrievability: `R = (1 + factor × t / S)^(-w20)` — 21 trained parameters
- Dual-strength model (Bjork & Bjork 1992): storage strength (grows) + retrieval strength (decays)
- Accessibility = retention×0.5 + retrieval×0.3 + storage×0.2
- 20-30% more efficient than SM-2 (Anki)

### 29 Cognitive Modules (stateful, persist across calls)

**Neuroscience (16):**
ActivationNetwork (Collins & Loftus 1975), SynapticTaggingSystem (Frey & Morris 1997), HippocampalIndex (Teyler & Rudy 2007), ContextMatcher (Tulving 1973), AccessibilityCalculator, CompetitionManager (Anderson 1994), StateUpdateService, ImportanceSignals, NoveltySignal, ArousalSignal, RewardSignal, AttentionSignal, EmotionalMemory (Brown & Kulik 1977), PredictiveMemory, ProspectiveMemory, IntentionParser

**Advanced (11):**
ImportanceTracker, ReconsolidationManager (Nader — 5min labile window), IntentDetector (9 intent types), ActivityTracker, MemoryDreamer (5-stage consolidation), MemoryChainBuilder (A*-like), MemoryCompressor (30-day min age), CrossProjectLearner (6 pattern types), AdaptiveEmbedder, SpeculativeRetriever (6 trigger types), ConsolidationScheduler

**Search (2):** Reranker, TemporalSearcher

### Memory States
- **Active** (retention ≥ 0.7) — easily retrievable
- **Dormant** (≥ 0.4) — retrievable with effort
- **Silent** (≥ 0.1) — difficult, needs cues
- **Unavailable** (< 0.1) — needs reinforcement

### Connection Types
semantic, temporal, causal, spatial, part_of, user_defined — each with strength (0-1), activation_count, timestamps

---

## Advanced Techniques

### Cross-Project Intelligence
The CrossProjectLearner tracks patterns across ALL projects (ErrorHandling, AsyncConcurrency, Testing, Architecture, Performance, Security). When you learn a pattern in one project that works, it becomes available in all projects. Use `codebase({ action: "get_context" })` without a codebase filter to get universal patterns.

### Reconsolidation Window
After any memory is accessed (via search, get, or promote), it enters a 5-minute "labile" state where modifications are enhanced. This is the optimal time to edit memories with new context. The system handles this automatically.

### Synaptic Tagging (Retroactive Importance)
Memories encoded in the last 9 hours can be retroactively promoted when something important happens. If you fix a critical bug, not only does the fix get saved — related memories from the past 9 hours also get importance boosts. The SynapticTaggingSystem handles this automatically.

### Dream Insights
Dreams don't just consolidate — they generate new insights by cross-referencing recent memories with older knowledge. The insights can reveal: contradictions between memories, previously unseen patterns, connections across different projects. Always check dream results for `insights_generated`.

### Token Budget Strategy
Use `token_budget` on search and session_context to control response size. For quick lookups: 500. For deep context: 3000-5000. Results that don't fit go to `expandable` — retrieve them with `memory({ action: "get", id: "..." })`.

### Detail Levels
- `brief` — id/type/tags/score only (1-2 tokens per result, good for scanning)
- `summary` — 8 fields including content preview (default, balanced)
- `full` — all FSRS state, timestamps, embedding info (for debugging/analysis)

---

## Memory Hygiene

**Promote** when user confirms helpful, solution worked, info was accurate.
**Demote** when user corrects mistake, info was wrong, led to bad outcome.
**Never save:** secrets, API keys, passwords, temporary debugging state, trivial info.

---

## The One Rule

**When in doubt, save. The cost of a duplicate is near zero (Prediction Error Gating handles dedup). The cost of lost knowledge is permanent.**

Memory is retrieval. Searching strengthens memory. Search liberally, save aggressively.

---

## Development

- **Crate:** `vestige-mcp` v2.0.4, Rust 2024 edition, MSRV 1.91
- **Tests:** 758 (406 mcp + 352 core), zero warnings
- **Build:** `cargo build --release -p vestige-mcp` (features: `embeddings` + `vector-search`)
- **Build (no embeddings):** `cargo build --release -p vestige-mcp --no-default-features`
- **Bench:** `cargo bench -p vestige-core`
- **Architecture:** `McpServer` → `Arc<Storage>` + `Arc<Mutex<CognitiveEngine>>`
- **Storage:** SQLite WAL mode, `Mutex<Connection>` reader/writer split, FTS5 full-text search
- **Embeddings:** nomic-embed-text-v1.5 (768D, 8K context) via fastembed (local ONNX, no API)
- **Vector index:** USearch HNSW (20x faster than FAISS)
- **Binaries:** `vestige-mcp` (MCP server), `vestige` (CLI), `vestige-restore`
- **Dashboard:** SvelteKit 2 + Svelte 5 + Three.js + Tailwind 4, embedded at `/dashboard`
- **Env vars:** `VESTIGE_DASHBOARD_PORT` (default 3927), `VESTIGE_HTTP_PORT` (default 3928), `VESTIGE_HTTP_BIND` (default 127.0.0.1), `VESTIGE_AUTH_TOKEN` (auto-generated), `VESTIGE_CONSOLIDATION_INTERVAL_HOURS` (default 6), `RUST_LOG`
