# Getting Started with Vestige

Your first 30 minutes, start to finish. By the end you'll have Vestige installed,
connected to your agent, storing memories, and you'll know how to look at exactly
what it kept.

Vestige is local-first: one binary, your data on your disk, no account, no cloud.

---

## 1. Install and connect (5 minutes)

Install is one command and connecting is one JSON block. The canonical, always-current
steps live in the README so this guide never drifts from them:

- **[Install + connect →](../README.md#-get-it-running-in-60-seconds)**
- Using a specific editor? Pick your per-agent guide —
  [Cursor](integrations/cursor.md), [VS Code](integrations/vscode.md),
  [Windsurf](integrations/windsurf.md), [JetBrains](integrations/jetbrains.md),
  [Xcode](integrations/xcode.md), [OpenCode](integrations/opencode.md),
  [Codex](integrations/codex.md) — or the
  **[Claude Desktop 2-minute setup](CONFIGURATION.md#claude-desktop-macos)**.

Confirm it's alive:

```bash
vestige-mcp --version     # prints the installed version
vestige stats             # 0 memories on a fresh install
```

---

## 2. What Vestige saves (and what it doesn't)

This is the thing most people get wrong on day one, so it's worth 60 seconds.

**Vestige does not record everything you type.** It is not a transcript logger. A
memory is written when:

- **You ask your agent to remember something** ("remember: we disable SimSIMD on
  release builds"), or
- **Your agent decides a fact is worth keeping** and calls the memory tool, or
- **You ingest deliberately** from the CLI: `vestige ingest "..."`.

Every write goes through **prediction-error gating** — near-duplicates of what you
already know are down-weighted, so the store doesn't fill with restatements of the
same fact. What you get is a curated set of durable facts, decisions, and the
occasional "this didn't work," not a firehose of your chat history.

If you want the deeper model (FSRS-6 decay, spreading activation, the science), see
**[The Science](SCIENCE.md)**. You don't need it to start.

---

## 3. Try the one thing nothing else does

The headline feature is a backward reach through time. Store a decision, then later
store a failure, and Vestige can surface the earlier decision as the *cause* — even
though the two share no words.

```bash
vestige ingest "We switched the prod cache from Redis to an in-memory LRU to cut costs."
# ...later...
vestige ingest "Prod is dropping sessions under load and users are getting logged out."

vestige backfill --contrast
```

`--contrast` shows you, side by side, what a plain similarity search returns versus
the real causal cause. That contrast is the whole pitch — more on the mechanism in
**[the README](../README.md#-the-thing-nothing-else-does-memory-with-hindsight)**.

You can also just talk to your agent normally; it will write and recall memories for
you through MCP. The CLI is for when you want to drive it directly.

---

## 4. Inspect what's stored

Vestige is built to be looked at. Nothing is hidden in an opaque index.

- **Quick counts:** `vestige stats`
- **Health check** (coverage, warnings): `vestige health`
- **Recall + reasoning** over your memories: `vestige recall "your question"`
- **Full export** to JSON/JSONL (grep it, diff it, back it up):
  `vestige export memories.jsonl --format jsonl`
- **The 3D dashboard** — watch your memory as a living graph:

  ```bash
  vestige dashboard          # opens http://localhost:3927
  ```

  (When running as the MCP server, enable it with `VESTIGE_DASHBOARD_ENABLED=1`.)

Because the store is a single SQLite database, you can also open it with any SQLite
browser if you want the raw truth.

---

## 5. Scope it: global vs per-project

By default Vestige uses one global memory in your OS data directory. For a
project-local memory that lives with the repo, point it at a directory:

```bash
vestige stats --data-dir ./.vestige          # this project's memory
VESTIGE_DATA_DIR=./.vestige vestige-mcp       # run the server against it
```

Precedence is `--data-dir` > `VESTIGE_DATA_DIR` > OS per-user default. Full details,
including multi-instance setups, are in **[Storage Modes](STORAGE.md)** and
**[Configuration](CONFIGURATION.md)**.

---

## Where to go next

| Want to… | Read |
|---|---|
| Understand a specific feature or the research | [The Science](SCIENCE.md) |
| Tune knobs, env vars, ports | [Configuration](CONFIGURATION.md) |
| Global vs per-project vs multi-instance | [Storage Modes](STORAGE.md) |
| Make your agent use memory automatically | [README → automatic memory](../README.md#-make-your-ai-use-memory-automatically) |
| Ask a real question | [FAQ](FAQ.md) |

Welcome. Vestige gets more useful the longer you use it — that's the point.
