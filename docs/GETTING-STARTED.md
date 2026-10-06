# Getting Started with Vestige

Your first 15 minutes on Vestige 4.x, start to finish. Every command on this page
was run against the 4.1.1 release binaries.

Vestige is the Causal Proof Engine and the operating system for AI agents. Its
kernel is Strata, an append-only, hash-chained, signed log on your machine. It is
local-first: four small binaries, your data on your disk, no account, no cloud, and
nothing downloads on first start.

---

## 1. Install and connect (5 minutes)

Install is two commands and connecting is one line or one JSON block. The canonical
steps live in the README so this guide never drifts from them:

- **[Install + connect →](../README.md#install)**
- Using an editor? [Cursor](integrations/cursor.md), [VS Code](integrations/vscode.md),
  [Windsurf](integrations/windsurf.md).

Confirm it is alive:

```bash
vestige-mcp --version     # vestige-mcp 4.1.1
vestige stats             # Total Memories: 0 on a fresh install
```

If `vestige-mcp --version` prints an older number, an older install comes first on
your PATH. `which -a vestige-mcp` lists every copy in the order the shell finds them.
A running Vestige server keeps the build it started with until its app restarts, so
restart every agent after an upgrade.

---

## 2. The one rule: you find a memory by its handle

This is what most people get wrong on day one, so it is worth 60 seconds.

Vestige 4.x does not search by resemblance. There are no embeddings, no keyword
search and no ranking by lookalike text. A record is found by an **exact handle**:

- its id, such as `mem-0000000000000001`
- a unique prefix of that id, 8 or more characters
- an exact tag, such as `incident`

So **tag every record on purpose**. The tag is how you, or your agent, get it back.
Tags are case-sensitive: `incident` and `Incident` are different handles. A tag
returns every live record that carries it, from every scope.

Ids are log positions, so they are not consecutive. Two saves in a row can come
back as `mem-0000000000000001` and `mem-0000000000000005`.

Vestige also does not record everything you type. A record is written when you ask
your agent to save something, when your agent decides a fact is worth keeping, or
when you ingest from the terminal. Nothing is merged or dropped as a near duplicate
in 4.x. Saving the same text twice reinforces the first record with an echo (the
response says `decision: "reinforce"`) and never creates a twin; different text is
always a new record. `dedup scan` lists exact duplicates (identical content hash or
identical declared source) and nothing more.

---

## 3. Save, find, and prove it (5 minutes)

Save two records. Both carry tags:

```bash
vestige ingest "We switched the prod cache from Redis to an in-memory LRU to cut costs." --tags cache,decision
vestige ingest "Prod is dropping sessions under load." --tags cache,incident
```

Each prints a `Node ID`. Find a record by tag or by id:

```bash
vestige recall --handle incident
vestige recall --handle mem-0000000000000001
```

Free text is refused, and the refusal tells you which handles it found in your words:

```bash
vestige recall "why are sessions dropping"
# Error: similarity_disabled: free-text recall is not a Strata operation ... Recall by exact handle instead
```

A prefix that matches more than one id is refused with the candidates, so you can pass
a longer prefix or the full id.

Check the log itself. This reads every frame and verifies the hash chain:

```bash
vestige strata-verify "<your data directory>"
```

The data directory is printed in the server's startup log, on the line that starts
`Strata log initialized at`. On macOS it is `~/Library/Application Support/com.vestige.core`.
Other platforms are in [Storage](STORAGE.md#data-directory).

Every write returns a receipt. Through an agent, `receipt` with `action: "get"` and the
record's id shows what that write did, and `action: "replay"` rebuilds the state from the
log and reports any mismatch.

---

## 4. Two things to try next

**Pairs nobody has connected yet.** GhostLink lists records that no recorded edge
joins and asks a question about each pair. These are leads, not findings:

```bash
vestige compose --lens divergent
```

With the two records above, it pairs the cache decision with the session incident
and asks you to name the mechanism that would connect them. Neither record has a typed
edge yet, so the pair is marked `unmeasured`: a forced juxtaposition picked by a
deterministic sampler, with no invented score. The default `bridge` lens lists nothing
until records have typed edges, and says so.

**Walking back from a failure.** `causal-walk` starts at a record you name and walks
backward over edges the log recorded. It never guesses:

```bash
vestige causal-walk --logged-write mem-0000000000000005
```

On a new store this reports that no recorded causal edge leads upstream from the
record. That is the correct answer: nothing has linked the two yet. The walk only
follows edges that were written. Your agent writes one when it saves a record with
`links`, for example the failure with `{"kind": "derived_from", "to": "<decision id>"}`.
Each link is a recorded edge with its own receipt. After that, the walk from the failure
reaches the decision. `links` accepts `derived_from`, `evidence_of` and `closes`.

`vestige selftest` plants a known cause in a throwaway copy and checks that the walk
finds it. Your live store is only read.

> The v3 command `vestige backfill --contrast` is not available on a 4.x store. It
> joined records by shared names, which 4.x does not treat as evidence. `causal-walk`
> is its successor.

---

## 5. Use it from your agent

Once an agent is connected, talk to it normally and name the tag:

> "Save that this project uses React with TypeScript. Tag it `stack`."

In a new session:

> "Recall the `stack` tag from Vestige."

Every save your agent makes goes through the log's gate and gets a receipt. Ask the
agent to call `receipt` with `action: "get"` and the record's id to see it.

Several agents can run Vestige at once on one machine. The first one to start serves
the store and the others connect to it, so every agent reads and writes the same log
through one writer.

---

## 6. Inspect, back up, export

- **Counts:** `vestige stats`
- **Health:** `vestige health`
- **Backup**, which works while your agents are running: `vestige backup <new-folder>`
- **Export** to JSON or JSONL: `vestige export memories.jsonl --format jsonl`
- **Credential audit** of what is already stored: `vestige scan-secrets`
- **Dashboard:** `vestige dashboard`

While an agent's server holds the store, `backup`, `strata-verify` and `dashboard`
work from the terminal. The other commands open the log directly, so they exit with
an error that names the holder. Run them when no agent is running, or use the same
tools through your agent.

The store is a signed, append-only log in `log/` inside the data directory. It is not
a SQLite file.

---

## 7. One store for everything, or one per project

By default Vestige keeps one store in your OS data directory. For a store that
lives with a repo, point it at a directory:

```bash
vestige stats --data-dir ./.vestige          # this project's store
VESTIGE_DATA_DIR=./.vestige vestige-mcp       # run the server against it
```

Precedence is `--data-dir`, then `VESTIGE_DATA_DIR`, then the OS per-user default.

---

## Coming from v3

Point 4.x at your existing data directory and start it. The first launch imports
`vestige.db` into a new log and leaves the v3 file untouched. Read
**[Upgrading from v3](../README.md#upgrading-from-v3)** first: every v3 process must be
stopped before the switch.

## Where to go next

| Want to… | Read |
|---|---|
| Every tool and what it returns | [Tool contracts](TOOL-CONTRACTS.md) |
| Why recall takes a handle | [README → Recall by handle](../README.md#recall-by-handle-not-resemblance) |
| GhostLink lenses and weaving | [README → GhostLink](../README.md#ghostlink-the-negative-space) |
| Backups and restore | [README → Backups and export](../README.md#backups-and-export) |
| Environment variables and `vestige.toml` | [Configuration](CONFIGURATION.md) |
| Where files live | [Storage](STORAGE.md) |
