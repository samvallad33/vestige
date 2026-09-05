# Storage Configuration

> Global, per-project, and multi-agent setups

---

## Database Location

All memories are stored in a **single local SQLite file**:

| Platform | Database Location |
|----------|------------------|
| macOS | `~/Library/Application Support/com.vestige.core/vestige.db` |
| Linux | `~/.local/share/vestige/core/vestige.db` |
| Windows | `%APPDATA%\vestige\core\vestige.db` |

Override precedence:

1. `vestige-mcp --data-dir <path>`
2. `VESTIGE_DATA_DIR=<path>`
3. OS default shown above

`--data-dir` and `VESTIGE_DATA_DIR` both point to a **directory**, not the database file itself. Vestige creates the directory if it does not exist, expands a leading `~`, and stores the database at `<data-dir>/vestige.db`.

---

## Moving Memories Between Devices

For device-to-device migration, use a portable archive instead of the normal JSON export:

```bash
# On the source machine
vestige portable-export ~/Desktop/vestige-portable.json

# On the destination machine, before adding memories
vestige portable-import ~/Desktop/vestige-portable.json
```

Portable archives preserve raw Vestige storage rows: memory IDs, FSRS state, graph connections, suppression state, timestamps, audit history, and embedding blobs.

For one-time migration, keep the conservative empty-database import:

```bash
vestige portable-import ~/Desktop/vestige-portable.json
```

For cross-device sync, use merge mode or the file-backed sync command:

```bash
# Merge a portable archive into an existing database.
vestige portable-import ~/Dropbox/vestige/portable.json --merge

# Pull, merge, and push through a shared archive file.
vestige sync ~/Dropbox/vestige/portable.json
```

`vestige sync` uses the same pluggable portable-sync backend interface as the core library. v2.1.1 ships a file backend, which works with Dropbox, iCloud Drive, Syncthing, Git, network shares, or any folder-sync system. The merge algorithm applies delete tombstones, keeps newer local memories on timestamp conflicts, preserves stable IDs, rebuilds FTS after import, and writes the pushed archive atomically when the filesystem supports rename. v2.1.2 also carries non-content purge tombstones so a hard purge can sync without retaining the deleted memory text.

When using the MCP `export` tool with `format: "portable"`, Vestige writes the archive under the active data directory's `exports/` folder. The MCP `restore` tool only reads from that `exports/` or `backups/` folder by default; pass `allowAnyPath: true` only for a trusted local file you selected manually.

The regular `vestige export` / `vestige restore` path remains useful for human-readable backups, partial exports, and older files, but it re-ingests memory content and does not preserve every storage-level relationship.

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

To set a global override for all MCP launches that inherit your shell environment:

```bash
export VESTIGE_DATA_DIR="~/.vestige"
```

### Option 2: Per-Project Memory

Separate memory per codebase. Good for:
- Client work (keep memories isolated)
- Different coding styles per project
- Team environments

**MCP Client Setup:**

Add an MCP server entry to your client or project config:
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

If both `VESTIGE_DATA_DIR` and `--data-dir` are set, the CLI flag wins. Use the env var for a machine-wide default and the CLI flag for per-client or per-project overrides.

The `vestige` CLI also honors `VESTIGE_DATA_DIR`, so use the same directory when inspecting or exporting a custom MCP instance:

```bash
VESTIGE_DATA_DIR=./.vestige vestige stats
VESTIGE_DATA_DIR=./.vestige vestige portable-export ./vestige-portable.json
```

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

### Option 3: Multi-Agent Household

For setups with multiple MCP clients or agent personas:

**Shared Memory (all clients share memories):**
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

**Separate Identities (each agent has its own memory):**

Client config for "Research":
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "~/vestige-research"]
    }
  }
}
```

Client config for "Builder":
```json
{
  "mcpServers": {
    "vestige": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "~/vestige-builder"]
    }
  }
}
```

---

## Data Safety

**Important:** Vestige stores data locally. v2.1.1 adds user-controlled file-backed sync through `vestige sync`, but Vestige does not run a hosted cloud service, background replication daemon, or automatic backup for you.

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

### The vector index follows peer writes

Each process holds its own in-memory HNSW index, rebuilt from
`embedding_profile_vectors` at startup. Before every semantic search it reads
`PRAGMA data_version`, which SQLite bumps when another connection commits, so
the check costs one pragma when nothing changed. Since schema V32 it also learns
*what* changed: every insert, update and delete on the vector table is journaled
by trigger into `vector_journal` (ids only, keyed by an AUTOINCREMENT sequence
that is monotonic in commit order and never reused), and the index applies
exactly the journaled rows past its watermark. A sibling process's new memory,
re-embedding or purge is therefore visible on the next query without a restart.
The consolidation cycle prunes the journal to the newest 10,000 rows plus the
last seven days; a process whose watermark has fallen behind that horizon
reconciles its index against the table instead of trusting the journal.

### What works today

| Pattern | Status | Notes |
|---------|--------|-------|
| One `vestige-mcp` + one MCP client | **Supported** | The default case. Zero contention. |
| Multiple MCP clients, separate `--data-dir` | **Supported** | Each process owns its own DB file. No shared state. |
| Multiple MCP clients, **shared** `--data-dir`, **one** `vestige-mcp` | **Supported** | Clients talk to a single MCP process that owns the DB. Recommended for multi-agent setups. |
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
