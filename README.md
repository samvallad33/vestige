<p align="center">
<pre>
██╗   ██╗███████╗███████╗████████╗██╗ ██████╗ ███████╗
██║   ██║██╔════╝██╔════╝╚══██╔══╝██║██╔════╝ ██╔════╝
██║   ██║█████╗  ███████╗   ██║   ██║██║  ███╗█████╗
╚██╗ ██╔╝██╔══╝  ╚════██║   ██║   ██║██║   ██║██╔══╝
 ╚████╔╝ ███████╗███████║   ██║   ██║╚██████╔╝███████╗
  ╚═══╝  ╚══════╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚══════╝
</pre>
</p>

<h1 align="center">Vestige</h1>

<p align="center">
  <strong>Memory traces that fade like yours do</strong>
</p>

<p align="center">
  The only AI memory system built on real cognitive science.<br/>
  FSRS-6 spaced repetition. Retroactive importance. Context-dependent recall.<br/>
  All local. All free.
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#features">Features</a> |
  <a href="#the-science">The Science</a>
</p>

<p align="center">
  <a href="https://github.com/samvallad33/vestige/releases"><img src="https://img.shields.io/github/v/release/samvallad33/vestige?style=flat-square" alt="Release"></a>
  <a href="https://github.com/samvallad33/vestige/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://github.com/samvallad33/vestige/actions"><img src="https://img.shields.io/github/actions/workflow/status/samvallad33/vestige/release.yml?style=flat-square" alt="Build"></a>
</p>

---

## Why Vestige?

**The only AI memory built on real cognitive science.**

| Feature | What It Does |
|---------|--------------|
| **FSRS-6 Spaced Repetition** | Full 21-parameter algorithm - nobody else in AI memory has this |
| **Retroactive Importance** | Mark something important, past 9 hours of memories strengthen too |
| **Context-Dependent Recall** | Retrieval matches encoding context (Tulving 1973) |
| **Memory States** | See if memories are Active, Dormant, Silent, or Unavailable |
| **100% Local** | No API keys, no cloud, your data stays yours |

> Other tools store memories. Vestige understands how memory actually works.

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/samvallad33/vestige
cd vestige
cargo build --release --package vestige-mcp
```

The binary will be at `./target/release/vestige-mcp`

### Homebrew (macOS/Linux)

```bash
brew install samvallad33/tap/vestige
```

---

## Quick Start

### 1. Build Vestige

```bash
cargo build --release --package vestige-mcp
```

### 2. Configure Claude Desktop

Add Vestige to your Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vestige": {
      "command": "/path/to/vestige-mcp",
      "args": [],
      "env": {
        "VESTIGE_DATA_DIR": "~/.vestige"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

Claude will now have access to persistent, biologically-inspired memory.

---

## Features

### Core

| Feature | Description |
|---------|-------------|
| **FSRS-6 Algorithm** | Full 21-parameter spaced repetition (20-30% better than SM-2) |
| **Dual-Strength Memory** | Bjork & Bjork (1992) - Storage + Retrieval strength model |
| **Hybrid Search** | BM25 + Semantic + RRF fusion for best retrieval |
| **Local Embeddings** | 768-dim BGE embeddings, no API required |
| **SQLite + FTS5** | Fast full-text search with persistence |

### Neuroscience-Inspired

| Feature | Description |
|---------|-------------|
| **Synaptic Tagging** | Retroactive importance (Frey & Morris 1997) |
| **Memory States** | Active/Dormant/Silent/Unavailable continuum |
| **Context-Dependent Memory** | Encoding specificity principle (Tulving 1973) |
| **Prospective Memory** | Future intentions with time/context triggers |
| **Basic Consolidation** | Decay + prune cycles |

### MCP Tools (26 Total)

**Core Memory (8):**
- `ingest` - Store new memories
- `smart_ingest` - Prediction Error Gating (auto-decides create/update/supersede)
- `recall` - Semantic retrieval
- `semantic_search` - Pure embedding search
- `hybrid_search` - BM25 + semantic fusion
- `get_knowledge` - Get memory by ID
- `delete_knowledge` - Remove memory
- `mark_reviewed` - FSRS review (1-4 rating)

**Stats & Maintenance (3):**
- `get_stats` - Memory statistics
- `health_check` - System health
- `run_consolidation` - Trigger consolidation

**Codebase Memory (3):**
- `remember_pattern` - Store code patterns
- `remember_decision` - Store architectural decisions
- `get_codebase_context` - Retrieve project context

**Prospective Memory (5):**
- `set_intention` - Remember to do something
- `check_intentions` - Check triggered intentions
- `complete_intention` - Mark intention done
- `snooze_intention` - Delay intention
- `list_intentions` - List all intentions

**Neuroscience (7):**
- `get_memory_state` - Check cognitive state
- `list_by_state` - Filter by state
- `state_stats` - State distribution
- `trigger_importance` - Retroactive strengthening
- `find_tagged` - Find strengthened memories
- `tagging_stats` - Tagging system statistics
- `match_context` - Context-dependent retrieval

---

## The Science

### Ebbinghaus Forgetting Curve (1885)

Memory retention decays exponentially over time:

```
R = e^(-t/S)
```

Where:
- **R** = Retrievability (probability of recall)
- **t** = Time since last review
- **S** = Stability (strength of memory)

### Bjork & Bjork Dual-Strength Model (1992)

Memories have two independent strengths:

- **Storage Strength**: How well encoded (never decreases)
- **Retrieval Strength**: How accessible now (decays with time)

Key insight: difficult retrievals increase storage strength more than easy ones.

### FSRS-6 Algorithm (2024)

Free Spaced Repetition Scheduler version 6. Trained on millions of reviews:

```rust
const FSRS_WEIGHTS: [f64; 21] = [
    0.40255, 1.18385, 3.173, 15.69105, 7.1949,
    0.5345, 1.4604, 0.0046, 1.54575, 0.1192,
    1.01925, 1.9395, 0.11, 0.29605, 2.2698,
    0.2315, 2.9898, 0.51655, 0.6621, 0.1, 0.5
];
```

### Synaptic Tagging & Capture (Frey & Morris 1997)

When something important happens, it can retroactively strengthen memories from the past several hours. Vestige implements this with a 9-hour capture window.

### Encoding Specificity Principle (Tulving 1973)

Memory retrieval is most effective when the retrieval context matches the encoding context. Vestige scores memories by context match.

---

## Comparison

| Feature | Vestige | Mem0 | Zep | Letta |
|---------|--------|------|-----|-------|
| FSRS-6 spaced repetition | Yes | No | No | No |
| Dual-strength memory | Yes | No | No | No |
| Retroactive importance | Yes | No | No | No |
| Memory states | Yes | No | No | No |
| Local embeddings | Yes | No | No | No |
| 100% local | Yes | No | No | No |
| Free & open source | Yes | Freemium | Freemium | Yes |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VESTIGE_DATA_DIR` | Data storage directory | `~/.vestige` |
| `VESTIGE_LOG_LEVEL` | Log verbosity | `info` |

---

## Development

### Prerequisites

- Rust 1.75+

### Building

```bash
git clone https://github.com/samvallad33/vestige
cd vestige
cargo build --release --package vestige-mcp
```

### Testing

```bash
cargo test --workspace
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

MIT OR Apache-2.0

---

<p align="center">
  <sub>Built with cognitive science and Rust.</sub>
</p>
