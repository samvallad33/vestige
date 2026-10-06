# Storage

> The Strata log: where Vestige keeps its state, how to place it, and how to back it up

Vestige's kernel is **Strata**, an append-only, hash-chained, signed log. Every write
is proposed, checked by the gate, and admitted as an effect. The state you read is
re-derived by replaying the log. Nothing in the shipped 4.x binaries links SQLite.

---

## Data directory

All state lives in one **data directory**. The log is the `log/` folder inside it.

| Platform | Default data directory |
|----------|------------------------|
| macOS | `~/Library/Application Support/com.vestige.core` |
| Linux | `$XDG_DATA_HOME/core`, or `~/.local/share/core` when that is unset |
| Windows | `%APPDATA%\vestige\core\data` |

The macOS path was checked against a running 4.1.1 install. The Linux and Windows
paths come from the `directories` crate (6.0.0) as `vestige-mcp` calls it with the
qualifier `com`, organization `vestige` and application `core`. Older docs showed
`~/.local/share/vestige/core` and `%APPDATA%\vestige\core`. If you are unsure, the
server's startup log names the directory on the line that starts `Strata log
initialized at`.

Override precedence:

1. `vestige-mcp --data-dir <path>` (or `vestige <command> --data-dir <path>`)
2. `VESTIGE_DATA_DIR=<path>`
3. The OS default above

Both point to a **directory**. Vestige creates it if it does not exist, expands a
leading `~`, and on Unix creates it owner-only (`0700`).

---

## What is in the data directory

| Path | What it is |
|------|------------|
| `log/` | The Strata log: segment files (`*.seg`), `head.state`, `strata.key` (the log's verifying key) and `strata.lock`. A sealed segment ends in a signed trailer |
| `store.meta` | Store metadata, present on some stores. `vestige backup` copies it with `log/` |
| `actor.key` | The identity key this process signs endorsements with. Backups do not include it |
| `receipt-signing.key` | Present after a v3 upgrade. `strata-verify` needs it beside the log to trust the migration receipt. Backups do not include it |
| `.serve.lock` | The OS lock the process that serves the store holds. A crashed holder releases it |
| `.serve.sock`, `.serve.endpoint` | While a server runs: the owner-only socket other agents attach to (loopback TCP on Windows) and the token file for it |
| `backups/` | `maintain` action `backup` writes `vestige-<time>.strata` folders here |
| `exports/` | `maintain` action `export` writes `memories-<time>.json` or `.jsonl` here |
| `maintenance-stamps.json` | The time of the last completed backup, which `session_start` reads for `needsBackup` |
| `vestige.toml` | Optional output defaults. See [Configuration](CONFIGURATION.md#output-configuration-vestigetoml) |
| `vestige.db` | Only after an upgrade from v3: the old file. 4.x never opens or modifies it |
| `upgrade.log` | Only after an upgrade from v3: what the importer did |

In the default per-user data directory, `auth_token` also lives here when you turn on
the HTTP transport. See [Dashboard access](CONFIGURATION.md#dashboard-access).

Check the log at any time, even while a server holds it:

```bash
vestige strata-verify "<data directory>"
```

It reads every frame, checks the hash chain and the signed trailers, and checks the
migration receipt if there is one. It creates no files.

---

## How a write reaches the log

1. A tool call **proposes** a write.
2. The **gate** checks it. A write that fails the gate, such as one carrying a live
   credential, is refused and writes nothing.
3. The write is **admitted** as an effect, and the call returns an `eff-` receipt.
4. State is derived by folding the log. `receipt get` shows what a write did.
   `receipt replay` rebuilds the state from the log and reports any mismatch.

Reads never append to the log. Retrievability, the FSRS value that says how far a
record has faded, is derived on read from the record's card and the clock. It is never
stored.

---

## Storage modes

### Option 1: One shared store (default)

One store for all projects. Good for personal preferences that apply everywhere and
for cross-project learning.

```bash
# Default behavior - no configuration needed
claude mcp add vestige vestige-mcp -s user
```

Records carry a **scope** (default `user`), so one store can still keep projects apart.
Exact-handle reads such as an id or a prefix work across scopes. Walks and GhostLink
proposals stay inside one scope unless you ask otherwise.

To set a machine-wide data directory for every launch that inherits your shell:

```bash
export VESTIGE_DATA_DIR="$HOME/.vestige"
```

### Option 2: Per-project store

A separate store per codebase. Good for client work you need to keep isolated.

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

This creates the log under `.vestige/` in your project root. Add `.vestige/` to
`.gitignore`. If both `VESTIGE_DATA_DIR` and `--data-dir` are set, the flag wins.

The `vestige` CLI honors the same directory:

```bash
VESTIGE_DATA_DIR=./.vestige vestige stats
vestige export ./memories.jsonl --format jsonl --data-dir ./.vestige
```

For both a global store and a project store, run two named servers:

```json
{
  "mcpServers": {
    "vestige-global": { "command": "vestige-mcp" },
    "vestige-project": {
      "command": "vestige-mcp",
      "args": ["--data-dir", "./.vestige"]
    }
  }
}
```

### Option 3: Multi-agent household

**Several agents, one store.** This is the default and needs no setup. Claude Code in
three terminals, Cursor, Codex and Claude Desktop can all run `vestige-mcp` against the
same data directory. The first process to start takes `.serve.lock` and serves the
store. Every later one connects to it through the owner-only socket and relays its
client's stdio there, so there is one writer. If the serving process quits, another
takes the lock and keeps serving, replaying its client's handshake so that session
continues. A request another agent had in flight at that moment returns an error rather
than being resent, because it may already have taken effect.

**Separate identities.** Give each agent its own data directory:

```json
{ "mcpServers": { "vestige": { "command": "vestige-mcp", "args": ["--data-dir", "~/vestige-research"] } } }
```

```json
{ "mcpServers": { "vestige": { "command": "vestige-mcp", "args": ["--data-dir", "~/vestige-builder"] } } }
```

Two different data directories are two different stores with two different writers.

---

## One writer

The log has exactly one writer. Which process that is can change over time, but at any
moment one process holds the store.

- **Agents** attach to the holder. They never open the log themselves.
- **CLI commands that open the log directly** (`stats`, `health`, `recall`, `ingest`,
  `export`, `compose`, `causal-walk`, `forgotten-lesson`, `selftest`, `scan-secrets`,
  `project`, `gc`) run only while no Vestige server holds the store. If one does, they
  exit with an error that names the process.
- **Three commands work while a server holds the store**, because they ask the holder
  or only read: `vestige backup`, `vestige strata-verify` and `vestige dashboard`.

If a command says another process holds the store, find the holder:

```bash
lsof -p "$(lsof -t "<data directory>/.serve.lock")" | grep txt
```

---

## Backups

```bash
vestige backup ~/backups/vestige-2026-10-04
```

This seals the log and copies `log/` (plus `store.meta`, when present) into a new,
empty folder. It works while your agents run, because it asks their server for the
copy. In that case the server makes the copy in `<data directory>/backups/` and the
command copies it to the folder you named, so one copy stays in `backups/`. The copy is owner-only (folders `0700`, files `0600`). A log that fails
verification is refused rather than copied. `vestige.db` is never copied: after an
upgrade it is the old v3 file, not the live store.

`maintain` action `backup` does the same into `<data directory>/backups/`.

Backups do not include `actor.key` or `receipt-signing.key`. Keep those safe, because
`strata-verify` of a backup of an upgraded store needs `receipt-signing.key` beside the
backup's `log/`.

**Restore** is a directory copy:

1. Stop every Vestige client.
2. Copy the backup's `log/` (and `store.meta`, when present) over the data directory's `log/`.
3. Keep the data directory's own `actor.key` and `receipt-signing.key`.

`vestige restore <file.json>` is not a backup restore. It re-ingests an export as new
records with new ids, fresh review state and no edges. Portable archives, `vestige sync`
and `maintain` action `restore` are not available on a Strata log.

For a machine-wide safety net, back up the data directory with Time Machine, Windows
Backup or `rsync` while no server runs, or schedule `vestige backup` yourself.

### Export

`vestige export <file> --format json|jsonl` and `maintain` action `export` write every
live record as JSON or JSONL. An export is for reading and for re-ingesting. It does not
carry edges, receipts or review state.

---

## Data safety

Vestige stores data locally. There is no hosted service, no background replication
and no automatic off-machine copy. The log is unencrypted on disk, so use full-disk
encryption if the records are sensitive.

| Use case | Risk | Recommendation |
|----------|------|----------------|
| Agent decisions and lessons | Medium | Take periodic backups |
| Code decisions anchored to a repo | Medium | Backups, plus `codebase verify` after refactors |
| Sensitive or regulated data | High | Not designed for it. Vestige gives no compliance guarantees |

Secrets: the gate refuses AWS, GitHub, Slack, Google, Anthropic, OpenAI and Stripe live
keys in every stored field. `vestige scan-secrets` audits what is already in the log.
Stripe test keys are not blocked, and a bare `password: ...` line is not blocked. The
log is append-only, so a credential that reached it stays on disk. Rotate it.

**Deletion.** `purge`, `memory` action `purge` or `delete`, and `delete_knowledge`
return `unavailable_in_4_0`. `suppress` takes a record out of every read and keeps its
bytes. Real erasure is planned as crypto-erasure ([#402](https://github.com/samvallad33/vestige/issues/402)).

---

## Inspecting the log

There is no SQL to run. Use the tools or the CLI:

| To see | Use |
|--------|-----|
| Counts and health | `memory_status` views `stats`, `health`, `retention`, or `vestige stats`, `vestige health` |
| One record and its origin frame | `memory` action `get`, `memory_status` view `provenance` with `memoryId` |
| What a write did | `receipt` action `get` with an `eff-` id or a record id |
| Whether the log still says what a receipt says | `receipt` action `replay` |
| The log's integrity | `vestige strata-verify <data directory>` |
| Records by day | `memory_status` view `timeline` |

---

## Moving to another machine

Copy the data directory while no Vestige process runs, or take a `vestige backup` and
restore it as above on the other machine. Keep `actor.key` and `receipt-signing.key`
with it if you want the same identity and a verifiable migration receipt.

---

## v3 storage (not used by the 4.x default build)

v3 kept everything in one SQLite file, `vestige.db`, with a vector index beside it,
and moved memories between devices with portable archives and `vestige sync`. The
4.x default build links no SQLite and has no vector index, so none of that applies
to a Strata log. The first 4.x launch on a v3 data directory imports `vestige.db`
into a new log through `vestige-upgrade`, verifies it against a signed migration
receipt, and leaves the v3 file byte-identical. See
[Migrating to Vestige 4.0](MIGRATING-v4.md) and
[Upgrading from v3](../README.md#upgrading-from-v3).
