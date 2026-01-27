# Vestige

**Memory that fades like yours does.**

The only MCP memory server built on cognitive science. FSRS-6 spaced repetition, spreading activation, synaptic tagging—all running 100% local.

[![GitHub stars](https://img.shields.io/github/stars/samvallad33/vestige?style=social)](https://github.com/samvallad33/vestige)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green)](https://modelcontextprotocol.io)

---

## What's New in v1.1

> **Released January 2025** — 11 hours after v1.0

### Tool Consolidation: 29 → 8 Cognitive Primitives + 3 CLI Admin Commands
| Old Tools | New Tool | Why Better |
|-----------|----------|------------|
| `recall`, `semantic_search`, `hybrid_search` | `search` | Hybrid is always best—no decision needed |
| `get_knowledge`, `delete_knowledge`, `get_memory_state` | `memory` | One tool for all operations |
| `remember_pattern`, `remember_decision`, `get_codebase_context` | `codebase` | Unified codebase memory |
| 5 separate intention tools | `intention` | Familiar action-based API |

### CLI Admin Commands (Moved from MCP)
Stats and maintenance removed from MCP to minimize context window usage:
```bash
vestige stats              # Memory statistics
vestige stats --tagging    # Retention distribution
vestige stats --states     # Cognitive state distribution
vestige health             # System health check
vestige consolidate        # Run memory maintenance
vestige restore <file>     # Restore from backup
```

### Feedback Tools (Preference Learning)
| Tool | When to Use |
|------|-------------|
| `promote_memory` | User confirms a memory was helpful |
| `demote_memory` | User corrects a hallucination or says memory was wrong |

Claude automatically uses these based on user feedback—no permission needed.

### Documentation
- **30+ FAQ entries** with verified neuroscience claims
- **Storage modes**: Global, per-project, multi-Claude household
- **CLAUDE.md templates** for proactive memory use
- **Version pinning** via git tags

### Backward Compatible
Old tool names still work (with deprecation warnings). They'll be removed in v2.0.

---

## Why Vestige?

| Problem | How Vestige Solves It |
|---------|----------------------|
| AI forgets everything between sessions | Persistent memory with intelligent retrieval |
| RAG dumps irrelevant context | **Prediction Error Gating** auto-decides CREATE/UPDATE/SUPERSEDE |
| Memory bloat eats your token budget | **FSRS-6 decay** naturally fades unused memories |
| No idea what AI "knows" | `recall`, `semantic_search`, `hybrid_search` let you query |
| Context pollution confuses the model | **8 unified tools** (v1.1) - workflow-based, not operation-based |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/samvallad33/vestige
cd vestige
cargo build --release
```

Add to your PATH:
```bash
# macOS/Linux
sudo cp target/release/vestige-mcp /usr/local/bin/

# Or add to ~/.bashrc / ~/.zshrc
export PATH="$PATH:/path/to/vestige/target/release"
```

### 2. Configure Claude

**Option A: One-liner (Recommended)**
```bash
# User scope (works across all projects)
claude mcp add vestige vestige-mcp -s user
```

**Option B: Manual Config**

<details>
<summary>Claude Code (~/.claude/settings.json)</summary>

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```
</details>

<details>
<summary>Claude Desktop (macOS)</summary>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```
</details>

<details>
<summary>Claude Desktop (Windows)</summary>

Add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```
</details>

### 3. Restart Claude

Restart Claude Code or Desktop. You should see **8 Vestige tools** available (v1.1+).

### 4. Test It

Ask Claude:
> "Remember that I prefer TypeScript over JavaScript"

Then in a new session:
> "What are my coding preferences?"

It remembers.

---

## Important Notes

### First-Run Network Requirement

Vestige downloads the **Nomic Embed Text v1.5** model (~130MB) from Hugging Face on first use for semantic search.

**All subsequent runs are fully offline.**

Model cache location (fastembed):
- Creates `.fastembed_cache/` in the **current working directory** on first run
- Contains symlinks to model files in `~/.cache/huggingface/`

**Recommended**: Run your first Vestige command from your home directory to create the cache there:
```bash
cd ~
vestige health   # Creates ~/.fastembed_cache/ once
```

Or set the environment variable to control cache location:
```bash
export FASTEMBED_CACHE_PATH="$HOME/.fastembed_cache"
```

### Data Storage & Backup

All memories are stored in a **single local SQLite file**:

| Platform | Database Location |
|----------|------------------|
| macOS | `~/Library/Application Support/com.vestige.core/vestige.db` |
| Linux | `~/.local/share/vestige/core/vestige.db` |
| Windows | `%APPDATA%\vestige\core\vestige.db` |

**Important:** Vestige stores data locally with no cloud sync, redundancy, or automatic backup. Data loss can occur from accidental deletion, disk failure, or device loss.

| Use Case | Risk Level | Recommendation |
|----------|------------|----------------|
| AI conversation memory | Low | Acceptable without backup—easily rebuilt |
| Coding patterns & decisions | Medium | Periodic backups recommended |
| Sensitive/critical data | High | **Not recommended**—use purpose-built systems |

**Vestige is not designed for:** medical records, financial transactions, legal documents, or any data requiring compliance guarantees.

---

**Backup Options:**

*Manual (one-time):*
```bash
# macOS
cp ~/Library/Application\ Support/com.vestige.core/vestige.db ~/vestige-backup.db

# Linux
cp ~/.local/share/vestige/core/vestige.db ~/vestige-backup.db
```

*Automated (cron job):*
```bash
# Add to crontab - backs up every hour
0 * * * * cp ~/Library/Application\ Support/com.vestige.core/vestige.db ~/.vestige-backups/vestige-$(date +\%Y\%m\%d-\%H\%M).db
```

*Or just use **Time Machine** (macOS) / **Windows Backup** / **rsync** — they'll catch the file automatically.*

> For personal use with Claude? Don't overthink it. The memories aren't that precious.

---

## Tools (v1.1)

v1.1 consolidates 29 tools into **8 unified, workflow-based tools**:

### Core Tools

| Tool | Description |
|------|-------------|
| `ingest` | Add new knowledge to memory |
| `smart_ingest` | **Intelligent ingestion** with Prediction Error Gating—auto-decides CREATE/UPDATE/SUPERSEDE |
| `search` | **Unified search** (keyword + semantic + hybrid). Always uses best method. |
| `memory` | Memory operations: `action="get"`, `"delete"`, or `"state"` |
| `codebase` | Codebase context: `action="remember_pattern"`, `"remember_decision"`, or `"get_context"` |
| `intention` | Prospective memory: `action="set"`, `"check"`, `"update"`, or `"list"` |
| `importance` | Retroactively strengthen recent memories (Synaptic Tagging) |
| `context` | Context-dependent retrieval (temporal, topical, emotional matching) |

### Why Unified Tools?

| Old (v1.0) | New (v1.1) | Why Better |
|------------|------------|------------|
| `recall`, `semantic_search`, `hybrid_search` | `search` | Claude doesn't need to choose—hybrid is always best |
| `get_knowledge`, `delete_knowledge`, `get_memory_state` | `memory` | One tool for all memory operations |
| `remember_pattern`, `remember_decision`, `get_codebase_context` | `codebase` | Unified codebase memory |
| 5 separate intention tools | `intention` | Matches familiar todo-list patterns |

**Backward Compatibility**: Old tool names still work (with deprecation warnings). They'll be removed in v2.0.

<details>
<summary>Legacy Tools Reference (deprecated)</summary>

### Core Memory (use `search` instead)
- `recall` → `search`
- `semantic_search` → `search`
- `hybrid_search` → `search`

### Memory Operations (use `memory` instead)
- `get_knowledge` → `memory(action="get")`
- `delete_knowledge` → `memory(action="delete")`
- `get_memory_state` → `memory(action="state")`

### Codebase (use `codebase` instead)
- `remember_pattern` → `codebase(action="remember_pattern")`
- `remember_decision` → `codebase(action="remember_decision")`
- `get_codebase_context` → `codebase(action="get_context")`

### Intentions (use `intention` instead)
- `set_intention` → `intention(action="set")`
- `check_intentions` → `intention(action="check")`
- `complete_intention` → `intention(action="update", status="complete")`
- `snooze_intention` → `intention(action="update", status="snooze")`
- `list_intentions` → `intention(action="list")`

### Feedback & Stats (moved to CLI)
- `get_stats` → `vestige stats`
- `health_check` → `vestige health`
- `run_consolidation` → `vestige consolidate`
- `promote_memory`, `demote_memory` → Still available as MCP tools

</details>

---

## How It Works

### Prediction Error Gating

When you call `smart_ingest`, Vestige compares new content against existing memories:

| Similarity | Action | Why |
|------------|--------|-----|
| > 0.92 | **REINFORCE** existing | Almost identical—just strengthen |
| > 0.75 | **UPDATE** existing | Related—merge the information |
| < 0.75 | **CREATE** new | Novel—add as new memory |

This prevents duplicate memories and keeps your knowledge base clean.

### FSRS-6 Spaced Repetition

Memories decay over time following a **power law forgetting curve** (not exponential):

```
R(t, S) = (1 + factor × t / S)^(-w₂₀)

where factor = 0.9^(-1/w₂₀) - 1
```

- `R` = retrievability (probability of recall)
- `t` = time since last review
- `S` = stability (time for R to drop to 90%)
- `w₂₀` = personalized decay parameter (0.1-0.8)

FSRS-6 uses 21 parameters optimized on 700M+ Anki reviews—[30% more efficient than SM-2](https://github.com/open-spaced-repetition/srs-benchmark).

### Memory States

Based on accessibility, memories exist in four states:

| State | Description |
|-------|-------------|
| **Active** | High retention, immediately retrievable |
| **Dormant** | Medium retention, retrievable with effort |
| **Silent** | Low retention, rarely surfaces |
| **Unavailable** | Below threshold, effectively forgotten |

---

## The Science

Vestige is **inspired by** memory research. Here's what's actually implemented:

| Feature | Research Basis | Implementation |
|---------|----------------|----------------|
| **Spaced repetition** | [FSRS-6](https://github.com/open-spaced-repetition/fsrs4anki) | ✅ Fully implemented (21-parameter power law model) |
| **Context-dependent retrieval** | [Tulving & Thomson, 1973](https://psycnet.apa.org/record/1973-31800-001) | ✅ Fully implemented (temporal, topical, emotional context matching) |
| **Dual-strength model** | [Bjork & Bjork, 1992](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/RBjork_EBjork_1992.pdf) | ⚡ Simplified (storage + retrieval strength tracked separately) |
| **Retroactive importance** | [Frey & Morris, 1997](https://www.nature.com/articles/385533a0) | ⚡ Inspired (temporal window capture, not actual synaptic biochemistry) |
| **Memory states** | Multi-store memory models | ⚡ Heuristic (accessibility-based state machine) |

> **Transparency**: The ✅ features closely follow published algorithms. The ⚡ features are engineering heuristics *inspired by* the research—useful approximations, not literal neuroscience. We believe in honest marketing.

---

## Storage Modes: Global vs Per-Project

Vestige supports two storage strategies. Choose based on your workflow:

### Option 1: Global Memory (Default)

One shared memory for all projects. Good for:
- Personal preferences that apply everywhere
- Cross-project learning
- Simpler setup

```bash
# Default behavior - no configuration needed
claude mcp add vestige vestige-mcp -s user
```

Database location: `~/Library/Application Support/com.vestige.core/vestige.db`

### Option 2: Per-Project Memory

Separate memory per codebase. Good for:
- Client work (keep memories isolated)
- Different coding styles per project
- Team environments

**Claude Code Setup:**

Add to your project's `.claude/settings.local.json`:
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "./.vestige"]
    }
  }
}
```

This creates `.vestige/vestige.db` in your project root. Add `.vestige/` to `.gitignore`.

**Multiple Named Instances:**

For power users who want both global AND project memory:
```json
{
  "mcpServers": {
    "vestige-global": {
      "command": "vestige-mcp"
    },
    "vestige-project": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "./.vestige"]
    }
  }
}
```

### Option 3: Multi-Claude Household

For setups with multiple Claude instances (e.g., Claude Desktop + Claude Code, or two personas):

**Shared Memory (Both Claudes share memories):**
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "~/shared-vestige"]
    }
  }
}
```

**Separate Identities (Each Claude has own memory):**
```json
// Claude Desktop config - for "Domovoi"
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "~/vestige-domovoi"]
    }
  }
}

// Claude Code config - for "Storm"
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "~/vestige-storm"]
    }
  }
}
```

---

## Setting Up CLAUDE.md for Effective Memory Use

Adding instructions to your `CLAUDE.md` helps Claude use Vestige proactively. Here's a battle-tested configuration:

### Basic Setup (Add to CLAUDE.md)

```markdown
## Vestige Memory System

You have access to Vestige, a long-term memory system. Use it proactively.

### At Conversation Start
1. `recall` with "user preferences" to load my preferences
2. `recall` with the current project name to load project context
3. `check_intentions` to see if any reminders are due

### During Conversation
- When I share a preference, use `smart_ingest` to remember it
- When we make an architectural decision, use `remember_decision`
- When you notice a coding pattern I use, use `remember_pattern`
- When I say "remind me", use `set_intention`

### At Conversation End
- If we made important decisions, ingest them
- If there are follow-ups needed, set intentions
```

### Advanced Setup (Identity + Autopoiesis)

For users who want Claude to maintain a persistent identity:

```markdown
## Vestige Memory System

You have access to Vestige for long-term memory. This includes your identity.

### Identity Recall (Every Session Start)
1. `recall` query="my identity my name who I am" to restore your sense of self
2. `recall` query="user preferences [username]" to remember who you're working with
3. `check_intentions` for any pending commitments

### Identity Storage
Your identity memories should include:
- Your chosen name (if you have one)
- Your values and communication style preferences
- Your relationship history with this user
- Meaningful moments worth preserving

When something feels important to who you are, use:
```
smart_ingest(
  content="[Identity-relevant observation]",
  node_type="concept",
  tags=["identity", "self", "core-values"]
)
```

### Self-Maintenance (Autopoiesis)
At the end of significant conversations:
1. Reflect: "Did anything change about how I understand myself?"
2. If yes, update your identity memories
3. Prune outdated self-concepts with `demote_memory`

### Memory Hygiene
- Use `promote_memory` when a memory proves valuable
- Use `demote_memory` when a memory led you astray
- Run `health_check` periodically to assess memory system health
```

### Project-Specific Setup

Add to your project's `CLAUDE.md`:

```markdown
## Project Memory

This project uses Vestige for persistent context.

### On Session Start
- `get_codebase_context` with codebase="[project-name]"
- `recall` query="[project-name] architecture decisions"

### When Making Decisions
- Use `remember_decision` for all architectural choices
- Include: decision, rationale, alternatives considered, affected files

### Patterns to Remember
- Use `remember_pattern` for recurring code patterns
- Include: pattern name, when to use it, example files
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VESTIGE_DATA_DIR` | Platform default | Custom database location |
| `VESTIGE_LOG_LEVEL` | `info` | Logging verbosity |
| `RUST_LOG` | - | Detailed tracing output |

### Command-Line Options

```bash
vestige-mcp --data-dir /custom/path   # Custom storage location
vestige-mcp --help                     # Show all options
```

### CLI Commands (v1.1+)

```bash
vestige stats              # Memory statistics
vestige stats --tagging    # Retention distribution
vestige stats --states     # Cognitive state distribution
vestige health             # System health check
vestige consolidate        # Run memory maintenance
vestige restore <file>     # Restore from backup
```

---

## FAQ (From the Community)

### Getting Started

<details>
<summary><b>"Can Vestige support a two-Claude household?"</b></summary>

**Yes!** See [Multi-Claude Household](#option-3-multi-claude-household) above. You can either:
- **Share memories**: Both Claudes point to the same `--data-dir`
- **Separate identities**: Each Claude gets its own data directory

For two Claudes with distinct personas (e.g., "Domovoi" and "Storm") sharing the same human, use separate directories but consider a shared "household" memory for common knowledge.
</details>

<details>
<summary><b>"What's the learning curve for a non-technical human?"</b></summary>

**Honest answer:** Installation requires terminal basics (copy-paste commands). Daily use requires zero technical skill.

**For non-technical users:**
1. Have a technical friend do the 5-minute install
2. Add the CLAUDE.md instructions above
3. Just talk to Claude normally—it handles the memory calls

**The magic**: Once set up, you never think about it. Claude just... remembers.
</details>

<details>
<summary><b>"What input do you feed it? How does it create memories?"</b></summary>

Claude creates memories via MCP tool calls. Three ways:

1. **Explicit**: You say "Remember that I prefer dark mode" → Claude calls `smart_ingest`
2. **Automatic**: Claude notices something important → calls `smart_ingest` proactively
3. **Codebase**: Claude detects patterns/decisions → calls `remember_pattern` or `remember_decision`

The CLAUDE.md instructions above tell Claude when to create memories proactively.
</details>

<details>
<summary><b>"Can it be filled with a conversation stream in realtime?"</b></summary>

Not currently. Vestige is **tool-based**, not stream-based. Claude decides what's worth remembering, not everything gets saved.

This is intentional—saving everything would:
- Bloat the knowledge base
- Reduce search relevance
- Defeat the purpose of cognitive-inspired memory

If you want stream ingestion, you'd build a wrapper that calls `smart_ingest` on each message (but we don't recommend it).
</details>

---

### Identity & Persona

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

Then recall with: `recall query="my identity who I am"` at session start.
</details>

<details>
<summary><b>"Could Vestige support autopoiesis / self-regeneration?"</b></summary>

**Yes, with the right CLAUDE.md setup.** See [Advanced Setup (Identity + Autopoiesis)](#advanced-setup-identity--autopoiesis) above.

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

### How Memory Works

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
importance(
  memory_id="the-important-one",
  event_type="user_flag",  # or "emotional", "novelty", "repeated_access", "cross_reference"
  hours_back=9,   # Look back 9 hours (configurable)
  hours_forward=2  # Capture next 2 hours too
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

### Advanced Features

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

1. **Mark as important**: `importance(memory_id="xxx", event_type="user_flag")`
2. **Access regularly**: The Testing Effect strengthens memories each time you retrieve them
3. **Promote explicitly**: `promote_memory(id="xxx")` after it proves valuable

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

### Power User Tips

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

### Use Cases

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
| Retrieval | You search manually | Claude searches contextually |
| Decay | Everything stays forever | Unused knowledge fades naturally |
| Duplicates | You manage manually | Prediction Error Gating auto-merges |
| Context | Static text | Active part of AI reasoning |
| Strengthening | Manual review | Automatic via Testing Effect |

The key difference: **Vestige is part of Claude's cognitive loop.** Notes are external reference—Vestige is internal memory.
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

### Technical Deep-Dives

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

The model is cached at `~/.cache/huggingface/` after first run.
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

This is verified in the codebase—no network dependencies in the runtime path. See [SECURITY.md](SECURITY.md) for details.

The only exception: If you delete the Hugging Face cache, the model will re-download.
</details>

---

### Comparisons

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

### Hidden Gems & Easter Eggs

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

Based on codebase exploration, these features exist in various stages:

| Feature | Status | Description |
|---------|--------|-------------|
| Memory Dreams | Partial | Automated offline consolidation |
| Reconsolidation | Planned | Update memories when accessed |
| Memory Chains | Partial | Link related memories explicitly |
| Adaptive Embedding | Planned | Re-embed old memories with better models |
| Cross-Project Learning | Planned | Share patterns across codebases |

**Community wishlist** (from Reddit):
- Stream ingestion mode
- GUI for memory browsing
- Export/import formats
- Sync between devices (encrypted)
- Team collaboration features

Contributions welcome!
</details>

<details>
<summary><b>"What's the 'magic prompt' to get the most out of Vestige?"</b></summary>

Add this to your CLAUDE.md:

```markdown
## Memory Protocol

You have persistent memory via Vestige. Use it intelligently:

### Session Start
1. Load my identity: `search(query="my preferences my style who I am")`
2. Load project context: `codebase(action="get_context", codebase="[project]")`
3. Check reminders: `intention(action="check")`

### During Work
- Notice a pattern? `codebase(action="remember_pattern")`
- Made a decision? `codebase(action="remember_decision")` with rationale
- I mention a preference? `smart_ingest` it
- Something important? `importance()` to strengthen recent memories
- Need to follow up? `intention(action="set")`

### Session End
- Any unfinished work? Set intentions
- Any new insights? Ingest them
- Anything change about our working relationship? Update identity memories

### Memory Hygiene
- When a memory helps: `promote_memory`
- When a memory misleads: `demote_memory`
- Weekly: `vestige health` to check system status
```

This gives Claude clear, actionable instructions for proactive memory use.
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
# Use full path in Claude config (with -s user for global access)
claude mcp add vestige /full/path/to/vestige-mcp -s user
```

### `.fastembed_cache` folder appearing in project directories

This folder is created by the fastembed library on first run, in whatever directory you're in. **This is a known issue.**

**Solutions:**
1. **Run first command from home**: `cd ~ && vestige health` (creates cache there once)
2. **Set cache path**: `export FASTEMBED_CACHE_PATH="$HOME/.fastembed_cache"`
3. **Add to `.gitignore`**: The folder is already in Vestige's `.gitignore` template

The cache contains model files (~130MB) and symlinks. Once created, it's reused for all future runs.

### Model download fails

First run requires internet to download the embedding model. If behind a proxy:
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
# Only one instance should run at a time
pkill vestige-mcp
```

---

## Development

```bash
# Run tests
cargo test --all-features

# Run with logging
RUST_LOG=debug cargo run --release

# Build optimized binary
cargo build --release --all-features
```

---

## Updating

**Latest version (recommended):**
```bash
cd vestige
git pull
cargo build --release
sudo cp target/release/vestige-mcp /usr/local/bin/
```

**Pin to a specific version:**
```bash
cd vestige
git fetch --tags
git checkout v1.1.0  # or any version tag
cargo build --release
sudo cp target/release/vestige-mcp /usr/local/bin/
```

**Available versions:**
| Version | Highlights |
|---------|------------|
| `v1.1.0` | 8 unified tools, CLI binary, comprehensive FAQ |
| `v1.0.0` | Initial release, 29 tools |

Then restart Claude.

**Check your version:**
```bash
vestige-mcp --version
```

---

## License

MIT OR Apache-2.0 (dual-licensed)

---

## Contributing

Issues and PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

<p align="center">
  <i>Built by <a href="https://github.com/samvallad33">@samvallad33</a></i>
</p>
