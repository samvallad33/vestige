# Storage Configuration

> Global, per-project, and multi-Claude setups

---

## Database Location

All memories are stored in a **single local SQLite file**:

| Platform | Database Location |
|----------|------------------|
| macOS | `~/Library/Application Support/com.vestige.core/vestige.db` |
| Linux | `~/.local/share/vestige/core/vestige.db` |
| Windows | `%APPDATA%\vestige\core\vestige.db` |

---

## Storage Modes

### Option 1: Global Memory (Default)

One shared memory for all projects. Good for:
- Personal preferences that apply everywhere
- Cross-project learning
- Simpler setup

```bash
# Default behavior - no configuration needed
claude mcp add vestige vestige-mcp -s user
```

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

Claude Desktop config - for "Domovoi":
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "~/vestige-domovoi"]
    }
  }
}
```

Claude Code config - for "Storm":
```json
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

## Data Safety

**Important:** Vestige stores data locally with no cloud sync, redundancy, or automatic backup.

| Use Case | Risk Level | Recommendation |
|----------|------------|----------------|
| AI conversation memory | Low | Acceptable without backup—easily rebuilt |
| Coding patterns & decisions | Medium | Periodic backups recommended |
| Sensitive/critical data | High | **Not recommended**—use purpose-built systems |

**Vestige is not designed for:** medical records, financial transactions, legal documents, or any data requiring compliance guarantees.

---

## Backup Options

### Manual (one-time)

```bash
# macOS
cp ~/Library/Application\ Support/com.vestige.core/vestige.db ~/vestige-backup.db

# Linux
cp ~/.local/share/vestige/core/vestige.db ~/vestige-backup.db
```

### Automated (cron job)

```bash
# Add to crontab - backs up every hour
0 * * * * cp ~/Library/Application\ Support/com.vestige.core/vestige.db ~/.vestige-backups/vestige-$(date +\%Y\%m\%d-\%H\%M).db
```

### System Backups

Just use **Time Machine** (macOS) / **Windows Backup** / **rsync** — they'll catch the file automatically.

> For personal use with Claude? Don't overthink it. The memories aren't that precious.

---

## Direct SQL Access

The database is just SQLite. You can query it directly:

```bash
sqlite3 ~/Library/Application\ Support/com.vestige.core/vestige.db

# Example queries
SELECT content, retention_strength FROM knowledge_nodes ORDER BY retention_strength DESC LIMIT 10;
SELECT content FROM knowledge_nodes WHERE tags LIKE '%identity%';
SELECT COUNT(*) FROM knowledge_nodes WHERE retention_strength < 0.1;
```

**Caution**: Don't modify the database while Vestige is running.

---

## Multi-Process Safety

Vestige's SQLite configuration is tuned for **safe concurrent reads alongside a single writer**. Multiple `vestige-mcp` processes pointed at the same database file is a supported *read-heavy* pattern; concurrent heavy writes from multiple processes is **experimental** and documented here honestly.

### What's shipped

Every `Storage::new()` call executes these pragmas on both the reader and writer connection (`crates/vestige-core/src/storage/sqlite.rs`):

```sql
PRAGMA journal_mode = WAL;        -- readers don't block writers, writers don't block readers
PRAGMA synchronous  = NORMAL;     -- durable across app crashes, not across OS crashes
PRAGMA cache_size   = -64000;     -- 64 MiB page cache per connection
PRAGMA temp_store   = MEMORY;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;       -- wait 5s on SQLITE_BUSY before surfacing the error
PRAGMA mmap_size    = 268435456;  -- 256 MiB memory-mapped I/O window
PRAGMA journal_size_limit = 67108864;
PRAGMA optimize = 0x10002;
```

Internally the `Storage` type holds **separate reader and writer connections**, each guarded by its own `Mutex<Connection>`. Within a single process this means:

- Any number of concurrent readers share the read connection lock.
- Writers serialize on the writer connection lock.
- WAL lets readers continue while a writer commits — they don't block each other at the SQLite level.

### What works today

| Pattern | Status | Notes |
|---------|--------|-------|
| One `vestige-mcp` + one Claude client | **Supported** | The default case. Zero contention. |
| Multiple Claude clients, separate `--data-dir` | **Supported** | Each process owns its own DB file. No shared state. |
| Multiple Claude clients, **shared** `--data-dir`, **one** `vestige-mcp` | **Supported** | Clients talk to a single MCP process that owns the DB. Recommended for multi-agent setups. |
| CLI (`vestige` binary) reading while `vestige-mcp` runs | **Supported** | WAL makes this safe — queries see a consistent snapshot. |
| Time Machine / `rsync` backup during writes | **Supported** | WAL journal gets copied with the main file; recovery handles it. |

### What's experimental

| Pattern | Status | Notes |
|---------|--------|-------|
| **Two `vestige-mcp` processes** writing the same DB concurrently | **Experimental** | SQLite serializes writers via a lock; if contention exceeds the 5s `busy_timeout`, writes surface `SQLITE_BUSY`. No exponential backoff or inter-process coordination layer beyond the pragma. |
| External writers (another SQLite client holding a write transaction open) | **Experimental** | Same concern as above — the 5s window is the only safety net. |
| Corrupted WAL recovery after hard-kill | **Supported by SQLite** | WAL is designed for crash recovery, but we do not explicitly test the `PRAGMA wal_checkpoint(RESTART)` path under load. |

If you hit `database is locked` errors:

```bash
# Identify the holder
lsof ~/Library/Application\ Support/com.vestige.core/vestige.db

# Clean shutdown of all vestige processes
pkill -INT vestige-mcp
```

### Why the "Stigmergic Swarm" story is honest

Multi-agent coordination through a shared memory graph — where agents alter the graph and other agents later *sense* those changes rather than passing explicit messages — is a first-class pattern on the **shared `--data-dir` + one `vestige-mcp`** setup above. In that configuration, every write flows through a single MCP process: WAL gives readers (agents querying state) a consistent view while the writer commits atomically, and the broadcast channel in `dashboard/events.rs` surfaces each cognitive event (dream, consolidation, promotion, suppression, Rac1 cascade) to every connected client in real time. No inter-process write coordination is required because there is one writer.

Running two or more `vestige-mcp` processes against the same file is where "experimental" kicks in. For the swarm narrative, point every agent at one MCP instance — that's the shipping pattern.

### Roadmap

Things we haven't shipped yet, tracked for a future release:

1. **File-based advisory lock** (`fs2` / `fcntl`) to detect and refuse startup when another `vestige-mcp` already owns the DB, instead of failing later with a lock error.
2. **Retry with jitter on `SQLITE_BUSY`** in addition to the pragma's blocking wait.
3. **Load test**: two `vestige-mcp` instances hammering the same file with mixed read/write traffic, verifying zero corruption and bounded write latency.

Until those land, treat "two writer processes on one file" as experimental. For everything else on this page, WAL + the 5s busy timeout is the shipping story.
