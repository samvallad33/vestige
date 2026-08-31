# Building a Cognitive Memory System with FSRS-6 and Three.js -- What 130 Years of Neuroscience Taught Us About AI Memory

Your AI assistant does not remember anything.

Every conversation starts from zero. You explain your project structure, your preferences, the bug you fixed last Tuesday, the architectural decision you made last month. The context window is a goldfish bowl -- 200K tokens of short-term memory, then nothing. RAG systems bolt on a vector database and call it "memory," but what they actually build is a search engine. Search is not memory. Memory is a living system that decays, strengthens, connects, and dreams.

Vestige is an open-source Rust MCP server that gives AI agents persistent memory modeled on real neuroscience. Not metaphorical neuroscience. Actual published algorithms from Ebbinghaus (1885), Collins & Loftus (1975), Bjork & Bjork (1992), Frey & Morris (1997), and the FSRS-6 spaced repetition scheduler trained on 700 million Anki reviews.

77,840+ lines of Rust. 29 cognitive modules. 734 tests. Single binary deployment with an embedded SvelteKit dashboard. AGPL-3.0 licensed.

Here is how we built it.

---

## The Problem: Session Boundaries are Amnesia

Every AI conversation today has the same failure mode. The model is stateless. Context windows are large but finite, and they reset between sessions. The industry's answer has been Retrieval-Augmented Generation -- embed documents, stuff them into the prompt, let the model figure it out.

RAG works for document Q&A. It does not work for memory, and here is why:

1. **No forgetting curve.** Every chunk in a vector database has equal weight forever. A configuration snippet from six months ago has the same retrieval priority as the bug fix from yesterday.
2. **No consolidation.** Memories are never merged, connected, or synthesized. You get back isolated chunks, not understanding.
3. **No retroactive importance.** If you flag something as important today, RAG cannot go back and strengthen the memories from last week that suddenly matter.
4. **No surprise detection.** Every insert is treated the same. Duplicate information bloats the database. Contradictions pile up silently.

We wanted something different. We wanted memory that behaves like a brain -- where memories compete, strengthen through use, decay through neglect, and form connections during idle periods.

---

## The Solution: Treat Memory Like a Brain, Not a Database

Vestige implements a cognitive architecture with three core principles:

1. **Memories have a lifecycle.** They are born, strengthen after explicit positive feedback, decay over time, can be revived, and eventually fade below the retrieval threshold. This is FSRS-6.
2. **Storage is gated by novelty.** Not everything deserves to be remembered. Prediction Error Gating compares new information against existing memories and decides whether to create, update, merge, or supersede. This is the hippocampal bouncer.
3. **Feedback changes memory.** Search is audit-only; explicit positive feedback applies the Testing Effect only to memories that proved useful.

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  AI Agent (Claude, GPT, etc.)                       │
│  ↕ JSON-RPC over stdio (MCP protocol)               │
├─────────────────────────────────────────────────────┤
│  vestige-mcp          19 MCP tools                  │
│  ├── Axum HTTP server  (dashboard + WebSocket)      │
│  ├── CognitiveEngine   (29 stateful modules)        │
│  └── Tool handlers     (one file per tool)          │
├─────────────────────────────────────────────────────┤
│  vestige-core          cognitive algorithms          │
│  ├── fsrs/             FSRS-6 spaced repetition     │
│  ├── neuroscience/     10 modules (STC, spreading   │
│  │                     activation, hippocampal       │
│  │                     index, importance signals...) │
│  ├── search/           hybrid, HyDE, reranker,      │
│  │                     keyword, vector, temporal     │
│  ├── advanced/         11 modules (dreams, PE gate,  │
│  │                     chains, compression,          │
│  │                     cross-project learning...)    │
│  ├── embeddings/       fastembed (Nomic v1.5, 768d) │
│  └── storage/          SQLite + FTS5 + USearch HNSW │
└─────────────────────────────────────────────────────┘
```

The entire system compiles to a single binary. The SvelteKit dashboard is embedded at compile time using Rust's `include_dir!` macro. No external services. No cloud dependencies. Your memories live on your machine.

---

## Deep Dive: FSRS-6 -- A Power Law Forgetting Curve

The Free Spaced Repetition Scheduler (FSRS-6) is the mathematical backbone of Vestige. Where traditional systems store memories with static priority scores, every memory in Vestige has two dynamic properties: **stability** (how deeply encoded it is) and **difficulty** (how hard it is to retain). These evolve over time according to a 21-parameter model trained on 700 million real-world Anki reviews.

The core formula is the power forgetting curve:

```
R(t, S) = (1 + factor * t / S) ^ (-w20)

where factor = 0.9 ^ (-1 / w20) - 1
```

- `R` is retrievability -- the probability you can recall this memory right now
- `t` is elapsed time since last access (in days)
- `S` is stability -- the number of days for R to drop to 90%
- `w20` is the personalizable decay parameter (default 0.1542)

Why power law instead of exponential? Because Ebbinghaus was right in 1885, and the data from 700 million reviews confirms it: human forgetting follows a power curve, not an exponential one. Power law decay has a heavier tail -- memories hang around longer than exponential models predict, which matches real behavior.

Here is the actual Rust implementation:

```rust
pub fn retrievability_with_decay(stability: f64, elapsed_days: f64, w20: f64) -> f64 {
    if stability <= 0.0 { return 0.0; }
    if elapsed_days <= 0.0 { return 1.0; }

    let factor = 0.9_f64.powf(-1.0 / w20) - 1.0;
    let r = (1.0 + factor * elapsed_days / stability).powf(-w20);
    r.clamp(0.0, 1.0)
}
```

FSRS-6 also introduced three new parameters (w17, w18, w19) for **same-day reviews** -- a gap in earlier versions that caused instability when memories were accessed multiple times within 24 hours. In an AI agent context where the same memory might be retrieved dozens of times in one session, this matters enormously.

Each memory exists in one of four states based on its accessibility score:

| State | Accessibility | Behavior |
|-------|--------------|----------|
| **Active** | >= 70% | Immediately retrievable, surfaces in searches |
| **Dormant** | 40-70% | Retrievable with effort, lower search priority |
| **Silent** | 10-40% | Rarely surfaces, needs direct access to revive |
| **Unavailable** | < 10% | Below retrieval threshold, candidate for GC |

Memories are never hard-deleted. They fade. An explicit positive feedback signal can strengthen a memory back toward Active after it proves useful; merely showing it in search cannot.

---

## Deep Dive: Prediction Error Gating -- The Hippocampal Bouncer

The brain does not store everything it perceives. The hippocampus acts as a novelty filter, comparing incoming information against existing memories and only consolidating what is genuinely new. This is prediction error: the gap between what you expected and what you got.

Vestige implements this as `PredictionErrorGate`. When `smart_ingest` is called with new content:

1. The content is embedded into a 768-dimensional vector (Nomic Embed v1.5)
2. Existing memories are searched for candidates above a similarity threshold
3. Cosine similarity determines the prediction error: `PE = 1.0 - similarity`
4. A decision is made:

```rust
pub enum GateDecision {
    Create   { reason, prediction_error, related_memory_ids },
    Update   { target_id, similarity, update_type, prediction_error },
    Supersede { old_memory_id, similarity, supersede_reason, prediction_error },
    Merge    { memory_ids, avg_similarity, strategy },
}
```

The thresholds:

| Similarity | Decision | Rationale |
|-----------|----------|-----------|
| > 0.92 | **Reinforce** | Near-identical content. Strengthen existing memory. |
| > 0.75 | **Update/Merge** | Related content. Merge information into existing memory. |
| 0.70-0.75 + contradiction detected | **Supersede** | Correction. New content replaces outdated memory. |
| < 0.75 | **Create** | Novel content. Store as new memory. |

Contradiction detection uses heuristic NLP -- looking for negation patterns ("don't" vs. "do", "avoid" vs. "use") and correction phrases ("actually", "the right way", "should be"). This catches the common case where a user corrects earlier advice.

The result: you can call `smart_ingest` aggressively without worrying about duplicates. The gate handles deduplication, merging, and conflict resolution automatically. The cost of a false positive (saving something redundant) is near zero because the gate will catch it. The cost of a false negative (losing knowledge) is permanent.

---

## Deep Dive: HyDE Search -- Query Expansion Without an LLM

Hypothetical Document Embeddings (HyDE) is a technique from Gao et al. (2022) where you use an LLM to generate a hypothetical answer to a query, embed that hypothetical answer, and use it for vector search. The intuition: a hypothetical answer is closer in embedding space to the real answer than the raw question is.

Full HyDE requires an LLM call at search time. That is too slow for a local-first system with sub-50ms search targets. Vestige implements a zero-latency approximation:

1. **Intent classification.** The raw query is classified into one of six intents: Definition, HowTo, Reasoning, Temporal, Lookup, or Technical.

2. **Template expansion.** Based on the intent, 3-5 variant queries are generated:

```rust
QueryIntent::Definition => {
    variants.push(format!("{clean} is a concept that involves"));
    variants.push(format!("The definition of {clean} in the context of"));
    variants.push(format!("{clean} refers to a type of"));
}
```

3. **Centroid embedding.** All variants are embedded, and the centroid (average) of the embedding vectors is computed and L2-normalized.

4. **Broadened search.** The centroid embedding captures a wider semantic space than any single query, improving recall for conceptual and question-style queries.

This gives approximately 60% of full HyDE quality improvement with zero latency overhead. The embedding model (Nomic v1.5 running locally via fastembed) generates all variant embeddings in a single batch.

The search pipeline then runs seven stages:

1. **Overfetch** -- Pull 3x results from hybrid search (BM25 keyword + semantic vector)
2. **Rerank** -- Re-score by relevance using a cross-encoder-style reranker
3. **Temporal boost** -- Recent memories get a recency bonus
4. **Accessibility filter** -- FSRS-6 retention threshold gates results (Ebbinghaus curve)
5. **Context match** -- Tulving's encoding specificity (1973): match current context to encoding context
6. **Competition** -- Anderson's retrieval-induced forgetting (1994): winners strengthen, competitors weaken
7. **Spreading activation** -- Collins & Loftus (1975): activate related memories as a side effect

The pipeline is the critical differentiator, but retrieval itself remains audit-only. Only explicit positive feedback reshapes a memory's retention state.

---

## Deep Dive: Synaptic Tagging and Capture -- Retroactive Importance

This is the feature that no other AI memory system has.

In 1997, Frey and Morris published a landmark paper in Nature describing Synaptic Tagging and Capture (STC). The finding: weak stimulation creates a temporary "synaptic tag" at a synapse. If a strong stimulation occurs within a temporal window (up to 9 hours), Plasticity-Related Products (PRPs) are produced that can be "captured" by the tagged synapses, consolidating them to long-term storage.

Translation for AI: **memories can become important retroactively.**

You have a conversation with a coworker about their vacation plans. Trivial. Three hours later, you learn they are leaving the company. Suddenly that vacation conversation is important context. In a traditional memory system, the vacation memory has already been classified as low-priority and buried. With STC, the "leaving the company" event triggers a backward sweep that captures and promotes the vacation conversation.

Vestige implements this with a 9-hour backward window and a 2-hour forward window:

```rust
const DEFAULT_BACKWARD_HOURS: f64 = 9.0;
const DEFAULT_FORWARD_HOURS: f64 = 2.0;
```

When an importance event occurs (user explicitly flags something, a novelty spike is detected, or repeated access patterns emerge), the STC system sweeps for tagged memories within the capture window. Capture probability decays with temporal distance using one of four configurable decay functions (exponential, linear, power law, or logarithmic).

Different event types have different capture characteristics:

| Event Type | Base Strength | Capture Radius | Use Case |
|-----------|--------------|----------------|----------|
| UserFlag | 1.0 | 1.0x | "Remember this" |
| NoveltySpike | 0.9 | 0.7x (narrow) | High prediction error |
| EmotionalContent | 0.8 | 1.5x (wide) | Sentiment detection |
| RepeatedAccess | 0.75 | 1.2x | Pattern of retrieval |

Captured memories are grouped into **importance clusters** -- temporal neighborhoods of memories that collectively provide context around a significant moment. This models how biological memory works: you do not remember isolated facts, you remember episodes.

---

## Deep Dive: Memory Dreaming -- Offline Consolidation

During sleep, the hippocampus replays recent experiences and transfers consolidated memories to the neocortex. This process discovers hidden connections between memories, strengthens important patterns, and prunes weak connections.

Vestige simulates this with a 5-stage dream cycle:

```
Stage 1 - Replay:        Replay recent memories in chronological order
Stage 2 - Cross-reference: Compare all memory pairs for hidden connections
Stage 3 - Strengthen:     Reinforce connections that co-activate
Stage 4 - Prune:          Decay weak connections, remove below threshold
Stage 5 - Transfer:       Identify memories ready for semantic storage
```

The dreaming system maintains a `ConnectionGraph` -- a weighted bidirectional graph where edges represent discovered relationships between memories. Edges have strength (0.0 to 2.0) and decay over time (factor 0.95 per consolidation cycle). Connections below 0.1 strength are pruned.

During Phase 2, the system evaluates memory pairs and discovers connections via multiple signals:

```rust
pub enum DiscoveredConnectionType {
    Semantic,       // High embedding similarity (> 0.8)
    SharedConcept,  // 2+ shared tags
    Temporal,       // Created within 24 hours + similarity > 0.6
    Complementary,  // Moderate similarity, different angles
    CausalChain,    // Cause-effect relationship detected
}
```

Phase 3 generates synthesized insights -- new knowledge that emerges from combining existing memories:

```rust
pub enum InsightType {
    HiddenConnection,   // "X and Y are related in ways you didn't notice"
    RecurringPattern,   // "You keep encountering this theme"
    Generalization,     // "These specific cases suggest a general rule"
    Contradiction,      // "These two memories conflict"
    KnowledgeGap,       // "You know X and Z but not Y"
    TemporalTrend,      // "This topic has evolved over the past month"
    Synthesis,          // "Combining A + B + C yields new understanding"
}
```

Dreams are triggered automatically: at session start if more than 24 hours have passed since the last dream, or after every 50 memory saves. The consolidation scheduler also monitors activity patterns and runs during detected idle periods (30+ minutes of inactivity).

---

## The Dashboard: Three.js Makes Memory Visible

Memory is invisible by default. You cannot debug what you cannot see. Vestige includes an embedded dashboard that renders the memory graph as a 3D force-directed visualization powered by Three.js with WebGL bloom post-processing.

Every memory is a glowing sphere. Size maps to retention strength. Color maps to node type (fact, concept, decision, pattern, event). Opacity fades as memories decay. Edges represent discovered connections, with opacity proportional to connection weight.

The visualization is event-driven via WebSocket. The Axum HTTP server runs alongside the MCP stdio transport, broadcasting `VestigeEvent` variants to all connected dashboard clients:

```rust
pub enum VestigeEvent {
    MemoryCreated { id, content_preview, node_type, tags, timestamp },
    SearchPerformed { query, result_count, result_ids, duration_ms, timestamp },
    DreamStarted { memory_count, timestamp },
    DreamProgress { phase, memory_id, progress_pct, timestamp },
    ConnectionDiscovered { source_id, target_id, connection_type, weight, timestamp },
    RetentionDecayed { id, old_retention, new_retention, timestamp },
    Heartbeat { uptime_secs, memory_count, avg_retention, timestamp },
    // ... 13 event types total
}
```

On the frontend, each event type triggers a distinct visual effect:

- **MemoryCreated:** Particle spawn burst (60 particles expanding outward) + expanding shockwave ring
- **SearchPerformed:** Blue pulse ripple across all nodes
- **DreamStarted:** Purple wash, bloom intensity increases to 1.5, rotation slows
- **DreamProgress:** Individual memories light up as they are "replayed"
- **ConnectionDiscovered:** Golden flash line between two nodes
- **RetentionDecayed:** Red pulse on the decaying node
- **ConsolidationCompleted:** Golden shimmer across all nodes

The force-directed layout uses a Fibonacci sphere distribution for initial positions with repulsion-attraction dynamics: nodes repel each other (Coulomb's law), edges attract connected nodes (spring force), and a centering force prevents drift. The simulation runs for 300 frames then settles.

The entire dashboard ships inside the Vestige binary. No separate frontend deployment. No CDN. `include_dir!` embeds the SvelteKit build output at compile time, and Axum serves it with proper MIME types and cache headers:

```rust
static DASHBOARD_DIR: Dir<'_> =
    include_dir!("$CARGO_MANIFEST_DIR/../../apps/dashboard/build");
```

---

## Architecture: Why Rust, SQLite, and Local Embeddings

### Rust

Memory is infrastructure. It runs on every interaction, on every search, on every save. Latency matters. We need sub-50ms search over thousands of memories, with embedding generation, FSRS calculations, and seven-stage pipeline execution. Rust gives us zero-cost abstractions, fearless concurrency (the `CognitiveEngine` is `Arc<Mutex<CognitiveEngine>>` shared across async handlers), and compile-time guarantees that the 29 stateful cognitive modules do not have data races.

### SQLite + FTS5 + USearch

SQLite is the most deployed database in the world for a reason. WAL mode gives us concurrent reads alongside writes. FTS5 gives us BM25 keyword search with zero operational overhead. USearch provides a Rust-native HNSW index for approximate nearest-neighbor vector search. The entire memory store is a single file at `~/.vestige/vestige.db`.

### fastembed (Nomic Embed v1.5)

All embeddings run locally. The Nomic Embed v1.5 model produces 768-dimensional vectors, runs via ONNX Runtime, and is competitive with OpenAI's ada-002. The model is cached in the platform user cache directory after first download (~130MB), with `./.fastembed_cache` as a fallback. No API keys. No network calls during operation. Your memories never leave your machine.

### Performance

| Memories | Search Time | Memory Usage |
|----------|-------------|--------------|
| 100 | < 10ms | ~50MB |
| 1,000 | < 50ms | ~100MB |
| 10,000 | < 200ms | ~300MB |

---

## Results: 734 Tests and What They Cover

Vestige has 734 tests across the workspace (313 in vestige-core, 338 in vestige-mcp, plus e2e tests). Every cognitive module has dedicated test coverage:

- **FSRS-6 algorithm:** Retrievability monotonic decay, round-trip interval calculation, sentiment boost, same-day review stability, difficulty mean reversion, fuzzing determinism
- **Prediction Error Gating:** Empty candidates, near-identical reinforcement, demoted memory supersession, orthogonal content creation, contradiction detection, force-create/force-update intents
- **Synaptic Tagging:** Tag creation and capture, PRP triggering, weak event rejection, clustering, tag decay and cleanup, batch operations, capture window probability
- **Spreading Activation:** Network creation, edge addition, BFS propagation with decay, activation thresholds, edge reinforcement
- **Memory Dreaming:** Full dream cycle, tag similarity, connection graph CRUD, consolidation scheduling, activity tracking

### Comparison with Existing Approaches

| Feature | RAG (Pinecone/Chroma) | mem0 | Vestige |
|---------|----------------------|------|---------|
| Forgetting curve | No | No | FSRS-6 (21-param power law) |
| Duplicate detection | Manual | Basic | Prediction Error Gating |
| Retroactive importance | No | No | Synaptic Tagging & Capture |
| Retrieval strengthening | No | No | Explicit Testing Effect feedback |
| Dream consolidation | No | No | 5-stage sleep model |
| Query expansion | No | No | HyDE (template-based) |
| 3D visualization | No | No | Three.js + WebSocket |
| Local embeddings | Optional | Cloud | Always local (Nomic v1.5) |
| Single binary | No | No | include_dir! embedded dashboard |
| License | Proprietary/OSS | OSS | AGPL-3.0 |

---

## What We Learned

**Neuroscience is an engineering goldmine.** The literature on human memory is vast, detailed, and largely untapped by the AI systems community. Papers from the 1970s through 2000s describe algorithms that directly translate into code -- Collins & Loftus's spreading activation is literally a BFS with weighted edges and decay. FSRS-6 is a parameterized forgetting curve. STC is a temporal window query with capture probability.

**The Testing Effect needs evidence.** Vestige records retrieval without treating it as correctness. When an agent explicitly promotes a memory that proved useful, that feedback strengthens the right record without allowing stale-but-relevant results to self-reinforce.

**Prediction Error Gating eliminates the "save or not" problem.** The single hardest UX question in AI memory is: what should be saved? The answer from neuroscience is: whatever is surprising. PE Gating compares against existing knowledge and only stores what is genuinely novel. This eliminates both the "save everything" bloat and the "save nothing" amnesia.

**Dreams are not a gimmick.** Offline consolidation consistently discovers connections that real-time search misses. When you replay 50 memories and compare all pairs, patterns emerge that individual searches would never find. The insight generation is simple (tag overlap + temporal proximity + embedding similarity), but the results are surprisingly useful.

---

## What's Next

Vestige v1.9 is targeting autonomic features: a retention target system that automatically adjusts consolidation frequency, adaptive embedding model selection based on content type, and a proactive suggestion engine that surfaces relevant memories before you search for them.

Further out: emotional memory tagging via sentiment analysis (the amygdala module), multi-agent memory sharing (let your coding agent share memories with your research agent), and a training loop that personalizes the FSRS-6 weights to your individual forgetting curve.

Memory is the missing layer between context windows and persistent knowledge. We think treating it as a cognitive system -- not a database -- is the right approach.

---

Vestige is open source under AGPL-3.0 at [github.com/samvallad33/vestige](https://github.com/samvallad33/vestige).

### References

- Ebbinghaus, H. (1885). *Uber das Gedachtnis*. Duncker & Humblot.
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review*, 82(6), 407-428.
- Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old theory of stimulus fluctuation. In *From learning processes to cognitive processes: Essays in honor of William K. Estes* (Vol. 2, pp. 35-67).
- Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. *Psychological Review*, 80(5), 352-373.
- Frey, U., & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385, 533-536.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science*, 17(3), 249-255.
- Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 20(5), 1063-1087.
- Redondo, R. L., & Morris, R. G. M. (2011). Making memories last: the synaptic tagging and capture hypothesis. *Nature Reviews Neuroscience*, 12(1), 17-30.
- Gao, L., et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. *arXiv:2212.10496*.
- Ye, J., et al. (2024). FSRS-6: A spaced repetition algorithm based on free recall. *github.com/open-spaced-repetition*.
