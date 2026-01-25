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
  FSRS-6 spaced repetition. Prediction Error Gating. Context-dependent recall.<br/>
  29 MCP tools. 100% local. 100% free.
</p>

<p align="center">
  <a href="#quick-start-2-minutes">Quick Start</a> |
  <a href="#all-29-tools">All 29 Tools</a> |
  <a href="#the-science">The Science</a>
</p>

<p align="center">
  <a href="https://github.com/samvallad33/vestige/releases"><img src="https://img.shields.io/github/v/release/samvallad33/vestige?style=flat-square" alt="Release"></a>
  <a href="https://github.com/samvallad33/vestige/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://github.com/samvallad33/vestige/actions"><img src="https://img.shields.io/github/actions/workflow/status/samvallad33/vestige/release.yml?style=flat-square" alt="Build"></a>
</p>

---

## Quick Start (2 minutes)

### Step 1: Build & Install

```bash
git clone https://github.com/samvallad33/vestige
cd vestige
cargo build --release
sudo cp target/release/vestige-mcp /usr/local/bin/
```

That's it. Vestige is now globally available.

### Step 2: Add to Claude Code

Add to your `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp"
    }
  }
}
```

### Step 3: Add to Claude Desktop (Optional)

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": [],
      "env": {
        "VESTIGE_DATA_DIR": "~/.vestige"
      }
    }
  }
}
```

### Step 4: Restart & Verify

Restart Claude Code/Desktop. You should see 29 Vestige tools available.

---

## Why Vestige?

| Feature | What It Does |
|---------|--------------|
| **Prediction Error Gating** | Auto-decides create/update/supersede based on similarity |
| **FSRS-6 Spaced Repetition** | Full 21-parameter algorithm trained on millions of reviews |
| **Retroactive Importance** | Mark something important, past 9 hours of memories strengthen |
| **Context-Dependent Recall** | Retrieval matches encoding context (Tulving 1973) |
| **Memory States** | Active/Dormant/Silent/Unavailable accessibility model |
| **100% Local** | No API keys, no cloud, no data leaves your machine |

---

## All 29 Tools

### Core Memory (8 tools)

| Tool | Description |
|------|-------------|
| `ingest` | Store a new memory |
| `smart_ingest` | **Prediction Error Gating** - auto-decides CREATE/UPDATE/SUPERSEDE based on semantic similarity to existing memories |
| `recall` | Semantic search with keyword matching |
| `semantic_search` | Pure embedding-based similarity search |
| `hybrid_search` | BM25 + semantic + RRF fusion (best retrieval quality) |
| `get_knowledge` | Retrieve a specific memory by ID |
| `delete_knowledge` | Remove a memory |
| `mark_reviewed` | FSRS review with 1-4 rating (strengthens memory) |

### Feedback System (3 tools)

| Tool | Description |
|------|-------------|
| `promote_memory` | Thumbs up - memory led to good outcome, increase retrieval strength |
| `demote_memory` | Thumbs down - memory was wrong/unhelpful, decrease retrieval strength |
| `request_feedback` | Ask user if a memory was helpful after using it |

### Stats & Maintenance (3 tools)

| Tool | Description |
|------|-------------|
| `get_stats` | Memory system statistics (total nodes, retention, embeddings) |
| `health_check` | System health status |
| `run_consolidation` | Trigger memory consolidation cycle (decay, promote, embed) |

### Codebase Memory (3 tools)

| Tool | Description |
|------|-------------|
| `remember_pattern` | Store a code pattern or convention |
| `remember_decision` | Store an architectural decision with rationale |
| `get_codebase_context` | Retrieve patterns and decisions for current project |

### Prospective Memory (5 tools)

| Tool | Description |
|------|-------------|
| `set_intention` | Remember to do something in the future (time/context/event triggers) |
| `check_intentions` | Check if any intentions should trigger based on current context |
| `complete_intention` | Mark an intention as fulfilled |
| `snooze_intention` | Delay an intention |
| `list_intentions` | List all active intentions |

### Neuroscience (7 tools)

| Tool | Description |
|------|-------------|
| `get_memory_state` | Check cognitive state (Active/Dormant/Silent/Unavailable) |
| `list_by_state` | List memories filtered by cognitive state |
| `state_stats` | Distribution of memories across states |
| `trigger_importance` | Retroactively strengthen recent memories (Synaptic Tagging) |
| `find_tagged` | Find memories with high retention (tagged/strengthened) |
| `tagging_stats` | Synaptic tagging statistics |
| `match_context` | Context-dependent retrieval (Tulving's Encoding Specificity) |

---

## Prediction Error Gating

The `smart_ingest` tool implements neuroscience-inspired memory gating:

```
New memory: "The API uses JWT tokens"
                    ↓
         [Prediction Error Gate]
                    ↓
┌────────────────────────────────────────────┐
│ Found similar: "API uses OAuth"            │
│ Similarity: 0.82 | Prediction Error: 0.18  │
│                                            │
│ Decision: UPDATE existing memory           │
│ (Not creating duplicate)                   │
└────────────────────────────────────────────┘
```

**Thresholds:**
- `>0.92` similarity → **Reinforce** (near-identical, just strengthen)
- `>0.75` similarity → **Update/Merge** (related, combine information)
- `<0.75` similarity → **Create** (sufficiently different, new memory)
- Demoted memory + similar new → **Supersede** (replace bad with good)

This solves the "bad vs good similar memory" problem automatically.

---

## The Science

### FSRS-6 Algorithm (2024)

Free Spaced Repetition Scheduler version 6. Trained on 700M+ reviews:

```rust
const FSRS_WEIGHTS: [f64; 21] = [
    0.40255, 1.18385, 3.173, 15.69105, 7.1949,
    0.5345, 1.4604, 0.0046, 1.54575, 0.1192,
    1.01925, 1.9395, 0.11, 0.29605, 2.2698,
    0.2315, 2.9898, 0.51655, 0.6621, 0.1, 0.5
];
```

20-30% better retention than SM-2 (what Anki uses).

### Bjork & Bjork Dual-Strength Model (1992)

Memories have two independent strengths:

- **Storage Strength**: How well encoded (never decreases)
- **Retrieval Strength**: How accessible now (decays with time)

Key insight: difficult retrievals increase storage strength more than easy ones.

### Synaptic Tagging & Capture (Frey & Morris 1997)

When something important happens, it retroactively strengthens memories from the past several hours. Vestige implements this with a 9-hour capture window.

### Encoding Specificity Principle (Tulving 1973)

Memory retrieval is most effective when the retrieval context matches the encoding context. The `match_context` tool scores memories by context similarity.

### Ebbinghaus Forgetting Curve (1885)

Memory retention decays exponentially: `R = e^(-t/S)`

Where R = retrievability, t = time, S = stability.

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VESTIGE_DATA_DIR` | Data storage directory | `~/.vestige` |
| `VESTIGE_LOG_LEVEL` | Log verbosity (`error`, `warn`, `info`, `debug`, `trace`) | `info` |

### Data Location

All data is stored in `~/.vestige/` by default:
- `vestige.db` - SQLite database with FTS5
- `embeddings/` - Local embedding cache

---

## Development

### Prerequisites

- Rust 1.75+

### Building

```bash
git clone https://github.com/samvallad33/vestige
cd vestige
cargo build --release
```

### Testing

```bash
cargo test --workspace
```

### Installing Locally

```bash
sudo cp target/release/vestige-mcp /usr/local/bin/
```

---

## Updating

```bash
cd vestige
git pull
cargo build --release
sudo cp target/release/vestige-mcp /usr/local/bin/
```

Restart Claude Code/Desktop to pick up changes.

---

## Troubleshooting

### "vestige-mcp: command not found"

Make sure you copied the binary:
```bash
sudo cp target/release/vestige-mcp /usr/local/bin/
```

### "No tools showing in Claude"

1. Check your config file syntax (valid JSON)
2. Restart Claude Code/Desktop completely
3. Check logs: `VESTIGE_LOG_LEVEL=debug vestige-mcp`

### "Embeddings not generating"

First run downloads the embedding model (~100MB). Check your internet connection and wait for download to complete.

---

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## License

MIT OR Apache-2.0

---

<p align="center">
  <sub>Built with cognitive science and Rust.</sub>
</p>
