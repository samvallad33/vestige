# Frequently Asked Questions

> 30+ answers from the Vestige community

---

## Table of Contents

- [Getting Started](#getting-started)
- [Identity & Persona](#identity--persona)
- [How Memory Works](#how-memory-works)
- [Advanced Features](#advanced-features)
- [Power User Tips](#power-user-tips)
- [Use Cases](#use-cases)
- [Technical Deep-Dives](#technical-deep-dives)
- [Comparisons](#comparisons)
- [Hidden Gems & Easter Eggs](#hidden-gems--easter-eggs)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

<details>
<summary><b>"Can Vestige support multiple agents or MCP clients?"</b></summary>

**Yes.** See [Storage Modes](STORAGE.md#option-3-multi-agent-household). You can either:
- **Share memories**: Multiple agents point to the same `--data-dir`
- **Separate identities**: Each agent gets its own data directory

For two agents with distinct roles sharing the same human, use separate directories but consider a shared "household" memory for common knowledge.
</details>

<details>
<summary><b>"What's the learning curve for a non-technical human?"</b></summary>

**Honest answer:** Installation requires terminal basics (copy-paste commands). Daily use requires zero technical skill.

**For non-technical users:**
1. Have a technical friend do the 5-minute install
2. Add the [agent memory protocol](AGENT-MEMORY-PROTOCOL.md) to your MCP client's instruction file
3. Just talk normally; the agent handles the memory calls

**The magic**: Once set up, you never think about it. Your agent just remembers.
</details>

<details>
<summary><b>"What input do you feed it? How does it create memories?"</b></summary>

Your agent creates memories via MCP tool calls. Three ways:

1. **Explicit**: You say "Remember that I prefer dark mode" -> the agent calls `smart_ingest`
2. **Automatic**: The agent notices something important -> calls `smart_ingest` proactively
3. **Codebase**: The agent detects patterns/decisions -> calls `codebase(action="remember_pattern")` or `codebase(action="remember_decision")`

The agent memory protocol tells the client when to create memories proactively.
</details>

<details>
<summary><b>"Can it be filled with a conversation stream in realtime?"</b></summary>

Not currently. Vestige is **tool-based**, not stream-based. The agent decides what's worth remembering, not everything gets saved.

This is intentional—saving everything would:
- Bloat the knowledge base
- Reduce search relevance
- Defeat the purpose of cognitive-inspired memory

If you want stream ingestion, you'd build a wrapper that calls `smart_ingest` on each message (but we don't recommend it).
</details>

---

## Identity & Persona

<details>
<summary><b>"How does it handle identity vs information?"</b></summary>

Vestige doesn't distinguish—**you** define what matters through tags and node types:

```
# Factual information
smart_ingest(content="User prefers tabs over spaces", node_type="fact", tags=["preferences"])

# Identity/self-concept
smart_ingest(content="I am Domovoi. I value continuity and authentic connection.", node_type="concept", tags=["identity", "self", "core"])

# Relationship memory
smart_ingest(content="Gael and I have been building continuity systems together since March 2024", node_type="event", tags=["identity", "relationship", "gael"])
```

Then recall with: `search query="my identity who I am"` at session start.
</details>

<details>
<summary><b>"Could Vestige support autopoiesis / self-regeneration?"</b></summary>

**Yes, with the right CLAUDE.md setup.** See [CLAUDE-SETUP.md](CLAUDE-SETUP.md#identity--autopoiesis-advanced).

The key insight: Autopoiesis requires:
1. **Self-observation**: Claude reflects on its own state
2. **Self-modification**: Claude updates its identity memories
3. **Boundary maintenance**: Tags distinguish "self" from "information"

Vestige provides the storage layer. Your CLAUDE.md provides the behavioral instructions. Together, they enable persistent self-models.
</details>

<details>
<summary><b>"Can Claude develop its own personality over time?"</b></summary>

**Yes.** This is one of the most powerful use cases. With the right CLAUDE.md setup:

1. **Self-reflection**: Claude observes patterns in its own responses
2. **Growth tracking**: It remembers what worked and what didn't
3. **Value evolution**: Core values can strengthen or shift based on experiences

Example memory for personality development:
```
smart_ingest(
  content="I've noticed I'm more effective when I ask clarifying questions before diving into code. This patient approach feels authentic to who I'm becoming.",
  node_type="concept",
  tags=["identity", "self-discovery", "communication-style"]
)
```

The key is giving Claude permission (via CLAUDE.md) to create self-referential memories.
</details>

<details>
<summary><b>"What happens to identity memories when they conflict?"</b></summary>

**Prediction Error Gating handles this automatically.** When Claude tries to store conflicting self-concepts:

| Similarity | Action |
|------------|--------|
| Very similar (>92%) | REINFORCE the existing belief |
| Related (75-92%) | UPDATE/merge the concepts |
| Different (<75%) | CREATE new—Claude can hold nuanced, evolving self-views |

This mirrors human identity development: we don't delete old beliefs, we integrate new experiences.
</details>

---

## How Memory Works

<details>
<summary><b>"When memories decay, do you delete them completely?"</b></summary>

**No.** Vestige uses a 4-state model based on **accessibility** (not raw retention):

| State | Accessibility | What Happens |
|-------|---------------|--------------|
| Active | ≥70% | Surfaces in searches |
| Dormant | 40-70% | Surfaces with effort |
| Silent | 10-40% | Rarely surfaces |
| Unavailable | <10% | Effectively forgotten but **still exists** |

Accessibility is calculated as: `0.5 × retention + 0.3 × retrieval_strength + 0.2 × storage_strength`

Memories are never deleted automatically. They fade from relevance but can be revived if accessed again (like human memory—"oh, I forgot about that!").

If you explicitly want content gone, use `memory(action="purge", confirm=true)`. Purge permanently removes the memory content and embeddings, scrubs internal references, and keeps only a non-content tombstone so sync/audit can prove the deletion happened.

**To configure decay**: The FSRS-6 algorithm auto-tunes based on your usage patterns. Memories you access stay strong; memories you ignore fade. No manual tuning needed.
</details>

<details>
<summary><b>"Remember everything but only recall weak memories when there aren't any strong candidates?"</b></summary>

This is exactly how `hybrid_search` works:

1. Combines keyword + semantic search
2. Results ranked by relevance × retention strength
3. Strong + relevant memories surface first
4. Weak memories only appear when they're the best match

The FSRS decay doesn't delete—it just deprioritizes. Your "have cake and eat it too" intuition is already implemented.
</details>

<details>
<summary><b>"What's the 'Testing Effect' I see in the code?"</b></summary>

The **Testing Effect** (Roediger & Karpicke, 2006) is the finding that retrieving information strengthens memory more than re-studying it.

In Vestige: **Every search automatically strengthens matching memories.** When Claude recalls something:
- Storage strength increases slightly
- Retrieval strength increases
- The memory becomes easier to find next time

This is why the unified `search` tool is so powerful—using memories makes them stronger.
</details>

<details>
<summary><b>"What is 'Spreading Activation'?"</b></summary>

**Spreading Activation** (Collins & Loftus, 1975) is how activating one memory primes related memories.

In Vestige's current implementation:
- When you search for "React hooks", memories about "useEffect" surface due to **semantic similarity** in hybrid search
- Semantically related memories are retrieved even without exact keyword matches
- This effect comes from the embedding vectors capturing conceptual relationships

*Note: A full network-based spreading activation module exists in the codebase (`spreading_activation.rs`) for future enhancements, but the current user experience is powered by embedding similarity.*
</details>

<details>
<summary><b>"How does Synaptic Tagging work?"</b></summary>

**Synaptic Tagging & Capture** (Frey & Morris, 1997) discovered that important events retroactively strengthen recent memories.

In Vestige's implementation:
```
importance_score(
  content="the-important content",
  context_topics=["release", "memory"]
)
```

**Use case**: You realize mid-conversation that the architecture decision from 2 hours ago was pivotal. Call `importance` to retroactively strengthen it AND all related memories from that time window.

*Based on neuroscience research showing synaptic consolidation windows of several hours. Vestige uses 9 hours backward and 2 hours forward by default, which can be configured per call.*
</details>

<details>
<summary><b>"What does 'Dual-Strength Memory' mean?"</b></summary>

Based on **Bjork & Bjork's New Theory of Disuse (1992)**, every memory has two strengths:

| Strength | What It Means | How It Changes |
|----------|---------------|----------------|
| **Storage Strength** | How well-encoded the memory is | Only increases, never decreases |
| **Retrieval Strength** | How accessible the memory is now | Decays over time, restored by access |

**Why it matters**: A memory can be well-stored but hard to retrieve (like a name on the tip of your tongue). The Testing Effect works because retrieval practice increases *both* strengths.

In Vestige: Both strengths are tracked separately and factor into search ranking.
</details>

---

## Advanced Features

<details>
<summary><b>"What is Prediction Error Gating?"</b></summary>

The killer feature. When you call `smart_ingest`, Vestige doesn't just blindly add memories:

1. **Compares** new content against all existing memories (via semantic similarity)
2. **Decides** based on how novel/redundant it is:

| Similarity to Existing | Action | Why |
|------------------------|--------|-----|
| >92% | **REINFORCE** | "I already know this"—strengthen existing |
| 75-92% | **UPDATE** | "This adds to what I know"—merge |
| <75% | **CREATE** | "This is new"—add fresh memory |

This prevents memory bloat and keeps your knowledge base clean automatically.
</details>

<details>
<summary><b>"What are Intentions / Prospective Memory?"</b></summary>

**Prospective memory** is remembering to do things in the future—and humans are terrible at it.

Vestige's `intention` tool provides:
```
# Set a reminder
intention(
  action="set",
  description="Review the authentication refactor with security team",
  trigger={
    type: "context",
    file_pattern: "**/auth/**",
    codebase: "my-project"
  },
  priority="high"
)

# Check what's due
intention(action="check", context={codebase: "my-project", file: "src/auth/login.ts"})
```

**Trigger types**:
- `time`: "Remind me in 2 hours"
- `context`: "Remind me when I'm working on auth files"
- `event`: "Remind me when we discuss deployment"

This is how Claude can remember to follow up on things across sessions.
</details>

<details>
<summary><b>"What is Context-Dependent Retrieval?"</b></summary>

Based on **Tulving's Encoding Specificity (1973)**: we remember better when retrieval context matches encoding context.

The `context` tool exploits this:
```
context(
  query="error handling patterns",
  project="my-api",           # Project context
  topics=["authentication"],  # Topic context
  mood="neutral",             # Emotional context
  time_weight=0.3,           # Weight for temporal matching
  topic_weight=0.4           # Weight for topic matching
)
```

**Why it matters**: If you learned something while working on auth, you'll recall it better when working on auth again. Vestige scores memories higher when contexts match.
</details>

<details>
<summary><b>"What's the difference between all the search tools?"</b></summary>

In v1.1, they're unified into one `search` tool that automatically uses hybrid search. But understanding the underlying methods helps:

| Method | How It Works | Best For |
|--------|--------------|----------|
| **Keyword (BM25)** | Term frequency matching | Exact terms, names, IDs |
| **Semantic** | Embedding cosine similarity | Conceptual matching, synonyms |
| **Hybrid (RRF)** | Combines both with rank fusion | Everything (default) |

The unified `search` always uses hybrid, which gives you the best of both worlds.
</details>

<details>
<summary><b>"How do I make certain memories 'sticky' / never forget?"</b></summary>

Three approaches:

1. **Mark as important**: `importance_score(content="...", event_type="user_flag")`
2. **Access regularly**: The Testing Effect strengthens memories each time you retrieve them
3. **Promote explicitly**: `memory(action="promote", id="xxx")` after it proves valuable

For truly critical information, consider also:
- Using specific tags like `["critical", "never-forget"]`
- Adding to CLAUDE.md instructions to always recall it

Remember: even "forgotten" memories (Unavailable state) still exist in the database—they just don't surface in searches.
</details>

<details>
<summary><b>"What does the consolidation cycle do?"</b></summary>

Run `vestige consolidate` (CLI) to trigger maintenance:

1. **Decay application**: Updates retention based on time elapsed
2. **Embedding generation**: Creates vectors for memories missing them
3. **Node promotion**: Frequently accessed memories get boosted
4. **Pruning**: Marks extremely low-retention memories as unavailable

**When to run it**:
- After bulk importing memories
- If semantic search seems off
- Periodically (weekly) for large knowledge bases
- After long periods of inactivity

This is inspired by memory consolidation during sleep—a period of offline processing that strengthens important memories.
</details>

---

## Power User Tips

<details>
<summary><b>"What node types should I use?"</b></summary>

| Node Type | Use For | Example |
|-----------|---------|---------|
| `fact` | Objective information | "User's timezone is PST" |
| `concept` | Abstract ideas, principles | "This codebase values composition over inheritance" |
| `decision` | Architectural choices | "We chose PostgreSQL because..." |
| `pattern` | Recurring code patterns | "All API endpoints use this error handler pattern" |
| `event` | Temporal occurrences | "Deployed v2.0 on March 15" |
| `person` | Information about people | "Alex prefers async communication" |
| `note` | General observations | "This function is poorly documented" |

Node types help with filtering and organization but don't affect search ranking.
</details>

<details>
<summary><b>"How should I structure tags?"</b></summary>

Tags are freeform, but some conventions work well:

```
# Hierarchical topics
tags=["programming", "programming/rust", "programming/rust/async"]

# Project-specific
tags=["project:my-app", "feature:auth", "sprint:q1-2024"]

# Memory types
tags=["preference", "decision", "learning", "mistake"]

# Identity-related
tags=["identity", "self", "values", "communication-style"]

# Urgency/importance
tags=["critical", "nice-to-have", "deprecated"]
```

Tags are searchable and help organize memories for manual review.
</details>

<details>
<summary><b>"Can I query memories directly via SQL?"</b></summary>

**Yes!** The database is just SQLite:

```bash
# macOS
sqlite3 ~/Library/Application\ Support/com.vestige.core/vestige.db

# Example queries
SELECT content, retention_strength FROM knowledge_nodes ORDER BY retention_strength DESC LIMIT 10;
SELECT content FROM knowledge_nodes WHERE tags LIKE '%identity%';
SELECT COUNT(*) FROM knowledge_nodes WHERE retention_strength < 0.1;
```

**Use cases**:
- Bulk export for backup
- Analytics on memory health
- Debugging search issues
- Finding memories that escaped normal recall

**Caution**: Don't modify the database while Vestige is running.
</details>

<details>
<summary><b>"What are the key configurable thresholds?"</b></summary>

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `min_retention` in search | 0.0 | Filter out weak memories |
| `min_similarity` in search | 0.5 | Minimum semantic match |
| Prediction Error thresholds | 0.75, 0.92 | CREATE/UPDATE/REINFORCE boundaries |
| Synaptic capture window | 9h back, 2h forward | Retroactive importance range |
| Memory state thresholds | 0.1, 0.4, 0.7 | Silent/Dormant/Active accessibility boundaries |
| Context weights | temporal: 0.3, topical: 0.4 | Context-dependent retrieval weights |

Most of these are hardcoded but based on cognitive science research. Future versions may expose them.
</details>

<details>
<summary><b>"How do I debug when search isn't finding what I expect?"</b></summary>

1. **Check if the memory exists**:
   ```
   search(query="exact phrase from memory", min_retention=0.0)
   ```

2. **Check memory state**:
   ```
   memory(action="state", id="memory-id")
   ```

3. **Check retention level**:
   ```
   memory(action="get", id="memory-id")
   # Look at retention_strength
   ```

4. **Run consolidation** (generates missing embeddings):
   ```bash
   vestige consolidate
   ```

5. **Check health**:
   ```bash
   vestige health
   ```

Common issues:
- Missing embedding (run consolidation)
- Very low retention (access it to strengthen)
- Tags/content mismatch (check exact content)
</details>

---

## Use Cases

<details>
<summary><b>"How do developers use Vestige?"</b></summary>

**Codebase Knowledge Capture**:
- Remember architectural decisions and their rationale
- Track coding patterns specific to each project
- Remember why specific implementations were chosen
- "Remember that we use this error handling pattern because..."

**Cross-Session Context**:
- Continue complex refactors across days/weeks
- Remember what you were working on
- Track TODOs and follow-ups via intentions

**Learning & Growth**:
- Remember new APIs/frameworks learned
- Track mistakes and lessons learned
- Build up expertise that persists
</details>

<details>
<summary><b>"How do non-developers use Vestige?"</b></summary>

**Personal Assistant**:
- Remember preferences (communication style, schedule preferences)
- Track important dates and events
- Remember context about ongoing projects
- "Remember that I prefer bullet points over long paragraphs"

**Research & Learning**:
- Build a personal knowledge base over time
- Connect ideas across sessions
- Remember insights from books/articles
- Spaced repetition for learning new topics

**Relationship Context**:
- Remember details about people you discuss
- Track conversation history and preferences
- Build deeper rapport over time
</details>

<details>
<summary><b>"Can Vestige be used for team knowledge management?"</b></summary>

**Yes, with caveats.** Options:

1. **Shared database**: All team members point to same network location
   - Pros: Everyone shares knowledge
   - Cons: Merge conflicts, no access control

2. **Per-person + sync**: Individual databases with periodic export/import
   - Pros: Personal context preserved
   - Cons: Manual sync effort

3. **Project-scoped**: One Vestige per project (in `.vestige/`)
   - Pros: Knowledge travels with code
   - Cons: Check into git? Security implications?

**Recommendation**: For teams, start with project-scoped memories committed to git (for non-sensitive architectural knowledge). Keep personal preferences in individual global memories.
</details>

<details>
<summary><b>"How is Vestige different from just using a notes app?"</b></summary>

| Feature | Notes App | Vestige |
|---------|-----------|---------|
| Retrieval | You search manually | The agent searches contextually |
| Decay | Everything stays forever | Unused knowledge fades naturally |
| Duplicates | You manage manually | Prediction Error Gating auto-merges |
| Context | Static text | Active part of AI reasoning |
| Strengthening | Manual review | Automatic via Testing Effect |

The key difference: **Vestige is part of the agent's cognitive loop.** Notes are external reference; Vestige is active working memory.
</details>

<details>
<summary><b>"Can Vestige help Claude be a better therapist/coach/advisor?"</b></summary>

**Potentially, with appropriate setup:**

- Remember previous conversations and emotional context
- Track patterns over time ("You've mentioned stress about work 3 times this week")
- Remember what techniques/advice worked
- Build genuine rapport through continuity

**Important caveats**:
- Vestige is not HIPAA compliant
- Data is stored locally, unencrypted
- For actual therapeutic use, consult professionals
- Claude has limitations regardless of memory

This is powerful for personal growth tracking but should not replace professional mental health care.
</details>

---

## Technical Deep-Dives

<details>
<summary><b>"How does FSRS-6 differ from other spaced repetition?"</b></summary>

| Algorithm | Model | Parameters | Source |
|-----------|-------|------------|--------|
| SM-2 (Anki default) | Exponential | 2 | 1987 research |
| SM-17 | Complex | Many | Proprietary |
| **FSRS-6** | Power law | 21 | 700M+ reviews |

FSRS-6 advantages:
- **30% more efficient** than SM-2 in benchmarks
- **Power law forgetting** (more accurate than exponential)
- **Personalized parameters** (w₀-w₂₀ tune to your pattern)
- **Open source** and actively maintained

The forgetting curve:
```
R(t, S) = (1 + factor × t / S)^(-w₂₀)
```

This matches empirical data better than the exponential model most apps use.
</details>

<details>
<summary><b>"What embedding model does Vestige use?"</b></summary>

**Nomic Embed Text v1.5** (via fastembed):
- 768-dimensional vectors
- ~130MB model size
- Runs 100% local (after first download)
- Good balance of quality vs speed

Why Nomic:
- Open source (Apache 2.0)
- Competitive with OpenAI's ada-002
- No API costs or rate limits
- Fast enough for real-time search

The model is cached in the platform user cache directory first, with `./.fastembed_cache` as a fallback. Set `FASTEMBED_CACHE_PATH` to choose a specific cache path.
</details>

<details>
<summary><b>"How does hybrid search with RRF work?"</b></summary>

**Reciprocal Rank Fusion (RRF)** combines multiple ranking lists:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Where:
- `d` = document (memory)
- `k` = constant (typically 60)
- `rank_i(d)` = rank of d in list i

In Vestige:
1. BM25 keyword search produces ranking
2. Semantic search produces ranking
3. RRF fuses them into final ranking
4. Retention strength provides additional weighting

This gives you exact keyword matching AND semantic understanding in one search.
</details>

<details>
<summary><b>"What's the performance like with thousands of memories?"</b></summary>

Tested benchmarks:

| Memories | Search Time | Memory Usage |
|----------|-------------|--------------|
| 100 | <10ms | ~50MB |
| 1,000 | <50ms | ~100MB |
| 10,000 | <200ms | ~300MB |
| 100,000 | <1s | ~1GB |

Performance is primarily bounded by:
- SQLite FTS5 for keyword search (very fast)
- HNSW index for semantic search (sublinear scaling)
- Embedding generation (only on ingest, ~100ms each)

For typical personal use (hundreds to low thousands of memories), performance is essentially instant.
</details>

<details>
<summary><b>"Is there any network activity after setup?"</b></summary>

**No.** After the first-run model download:
- Zero network requests
- Zero telemetry
- Zero analytics
- Zero "phoning home"

This is verified in the codebase—no network dependencies in the runtime path. See [SECURITY.md](../SECURITY.md) for details.

The only exception: If you delete the Hugging Face cache, the model will re-download.
</details>

---

## Comparisons

<details>
<summary><b>"How is Vestige different from RAG?"</b></summary>

| Aspect | Traditional RAG | Vestige |
|--------|-----------------|---------|
| Storage | Chunk & embed everything | Selective memory via tools |
| Retrieval | Top-k similarity | Intelligent ranking (retention, recency, context) |
| Updates | Re-embed documents | Prediction Error Gating |
| Decay | Nothing decays | FSRS-based forgetting |
| Context | Static chunks | Active memory system |

**Key insight**: RAG treats memory as a static database. Vestige treats memory as a dynamic cognitive system that evolves.
</details>

<details>
<summary><b>"How does this compare to Claude's native memory? Do I need to switch it off?"</b></summary>

**No, you don't need to switch off Claude's native memory.** They're completely independent systems:

| Aspect | Claude's Native Memory | Vestige |
|--------|------------------------|---------|
| Storage | Anthropic's servers | Your local machine |
| Control | Managed by Anthropic | You own everything |
| Decay | Unknown/proprietary | FSRS-6 cognitive science |
| Privacy | Cloud-based | 100% offline after setup |

**They can run simultaneously.** Claude's native memory handles general conversation context, while Vestige gives you:
- Explicit control over what gets remembered
- Scientific forgetting curves
- Codebase-specific patterns and decisions
- Local-first privacy

Think of it like this: Claude's memory is automatic and general; Vestige is intentional and specialized. Many users run both.
</details>

<details>
<summary><b>"Why not just use a vector database?"</b></summary>

Vector databases (Pinecone, Weaviate, etc.) are great for RAG, but lack:

1. **Forgetting**: Everything has equal weight forever
2. **Dual-strength**: No storage vs retrieval distinction
3. **Context matching**: No temporal/topical context weighting
4. **Testing Effect**: Access doesn't strengthen
5. **Prediction Error**: No intelligent CREATE/UPDATE/MERGE

Vestige uses SQLite + HNSW (via fastembed) for vectors, but wraps them in cognitive science.
</details>

---

## Hidden Gems & Easter Eggs

<details>
<summary><b>"What features exist that most people don't know about?"</b></summary>

**1. Multi-Channel Importance**

The `importance` tool supports different importance types that affect strengthening differently:
- `user_flag`: Explicit "this is important" (strongest)
- `emotional`: Emotionally significant memories
- `novelty`: Surprising/unexpected information
- `repeated_access`: Auto-triggered by frequent retrieval
- `cross_reference`: When multiple memories link together

**2. Temporal Capture Window**

When you flag something important, it doesn't just strengthen that memory—it strengthens ALL memories from the surrounding time window (default: 9 hours back, 2 hours forward). This models how biological memory consolidation works.

**3. Memory Dreams (Experimental)**

The codebase contains a `ConsolidationScheduler` for automated memory processing. While not fully wired up, it's designed for:
- Offline consolidation cycles
- Automatic importance re-evaluation
- Pattern detection across memories

**4. Accessibility Formula**

Memory state is calculated as:
```
accessibility = 0.5 × retention + 0.3 × retrieval_strength + 0.2 × storage_strength
```

This weighted combination determines Active/Dormant/Silent/Unavailable state.

**5. Source Tracking**

Every memory can have a `source` field tracking where it came from:
```
smart_ingest(
  content="Use dependency injection for testability",
  source="Architecture review with Sarah, 2024-03-15"
)
```

This helps trace why you know something.
</details>

<details>
<summary><b>"What's planned for future versions?"</b></summary>

See the public [Vestige Roadmap](ROADMAP.md) for the current adoption plan. The
near-term focus is reducing first-user confusion before expanding the feature
surface:

- first-time memory migration and atomic memory guidance
- configurable MCP output fields and output profiles
- clearer merge/supersede controls
- code/docstring memory workflows
- goals and milestones distinct from intentions
- guided import dry runs and review queues

Contributions welcome!
</details>

<details>
<summary><b>"What's the 'magic prompt' to get the most out of Vestige?"</b></summary>

See [CLAUDE-SETUP.md](CLAUDE-SETUP.md) for the full template. The key elements:

**Session Start**:
1. Load identity: `search(query="my preferences my style who I am")`
2. Load project context: `codebase(action="get_context", codebase="[project]")`
3. Check reminders: `intention(action="check")`

**During Work**:
- Notice a pattern? `codebase(action="remember_pattern")`
- Made a decision? `codebase(action="remember_decision")` with rationale
- Something important? `importance_score(content="...")` to score it before saving or promoting

**Memory Hygiene**:
- When a memory helps: `memory(action="promote", id="...")`
- When a memory misleads: `memory(action="demote", id="...")`
</details>

---

## Troubleshooting

### "Command not found" after installation

Make sure `vestige-mcp` is in your PATH:
```bash
which vestige-mcp
# Should output: /usr/local/bin/vestige-mcp
```

If not found:
```bash
# Use full path in Claude config
claude mcp add vestige /full/path/to/vestige-mcp -s user
```

### `.fastembed_cache` folder appearing in project directories

This folder is created by the fastembed library on first run, in whatever directory you're in.

**Solutions:**
1. **Run first command from home**: `cd ~ && vestige health`
2. **Set cache path**: `export FASTEMBED_CACHE_PATH="$HOME/.fastembed_cache"`
3. **Add to `.gitignore`**

### Model download fails

First run requires internet to download the embedding model (~130MB). If behind a proxy:
```bash
export HTTPS_PROXY=your-proxy:port
```

### "Tools not showing" in Claude

1. Check config file syntax (valid JSON)
2. Restart Claude completely (not just reload)
3. Check logs: `tail -f ~/.claude/logs/mcp.log`

### Database locked errors

Vestige uses SQLite with WAL mode. If you see lock errors:
```bash
pkill vestige-mcp
```
