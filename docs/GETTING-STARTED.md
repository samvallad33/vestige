# Getting Started with Vestige

Your first 15 minutes on Vestige v6.2, start to finish. Every command on this page
runs against the v6.2.1 release binaries.

Vestige is local-first: an append-only, cryptographic Strata memory kernel, your data
on your disk, no account, no cloud, and sub-millisecond reflex hooks over a single
live memory graph.

---

## 1. Install and connect (5 minutes)

Install is two commands and connecting is one line or one JSON block:

- **[Install + connect →](REFERENCE.md#install)**
- Using an editor? [Cursor](integrations/cursor.md), [VS Code](integrations/vscode.md),
  [Windsurf](integrations/windsurf.md).

Confirm it is alive:

```bash
vestige-mcp --version     # vestige-mcp 6.2.1
vestige stats             # Total Memories: 0 on a fresh install
```

If `vestige-mcp --version` prints an older number, check `which -a vestige-mcp` to ensure
the newest install takes precedence on your PATH.

---

## 2. The 6 Cognitive Verbs

In v6.2, Vestige collapsed 18 disparate tools into 6 distinct cognitive verbs:

1. `session`: Session lifecycle boot, status, intentions, and lock-free rule projection.
2. `remember`: Single gated write, codebase anchoring, and inline Retroactive Salience Backfill.
3. `recall`: Exact-handle retrieval with structured candidate discovery (`discovered_handles`).
4. `causal_walk`: Root-cause backwards walk, forgotten lesson surfacing, and deterministic bisect proof cards (`action="prove"`).
5. `curate`: Memory state mutations, promotions, demotions, edits, bridges, and identity connections.
6. `precedent`: Operator guard stops, receipts, replay, BLAKE3/Ed25519 log verification, and secret scanning.

All legacy tool names remain supported as internal compatibility aliases.

---

## 3. The one rule: you find a memory by its handle

Vestige does not search by lossy embedding resemblance or keyword BM25. A memory is found by an **exact handle**:

- its id, such as `mem-0000000000000001`
- an exact tag, such as `incident`

So **tag every memory on purpose**. The tag is how you, or your agent, get it back.
Tags are case-sensitive.

When an agent passes an unindexed query, `recall` returns structured `discovered_handles: [...]`
so the model recovers immediately on the very next turn.

---

## 4. Save, find, and prove it (5 minutes)

Save two memories. The second one carries tags:

```bash
vestige ingest "We switched the prod cache from Redis to an in-memory LRU to cut costs." --tags cache,decision
vestige ingest "Prod is dropping sessions under load." --tags cache,incident
```

Find a memory by tag or by id:

```bash
vestige recall --handle incident
vestige recall --handle mem-0000000000000001
```

Check the cryptographic Strata log. This reads every frame and verifies the hash chain:

```bash
vestige strata-verify "<your data directory>"
```

The data directory is printed in the server's startup log. On macOS it is
`~/Library/Application Support/com.vestige.core`.

---

## 5. Walking back from a failure and proving causes

**Walking back from a failure.** `causal-walk` starts at a memory you name and walks
backward over edges the log recorded:

```bash
vestige causal-walk --logged-write mem-0000000000000005
```

**Proving boundaries.** Run deterministic bisect and generate a signed proof card:

```bash
vestige prove --logged-write mem-0000000000000005 --repo . --good main~5 --bad main --test "npm test"
```

---

## 6. Inspect, back up, export

- **Counts & Health:** `vestige stats`, `vestige health` (routed over zero-lock IPC)
- **Backup:** `vestige backup <new-folder>` (sealed Ed25519 segments)
- **Export:** `vestige export memories.jsonl --format jsonl`
- **Secret Scan:** `vestige scan-secrets`
- **Dashboard:** `vestige dashboard`

While `vestige-mcp` is running, CLI commands automatically route via `~/.vestige/reflex.sock`,
completely eliminating `.serve.lock` collisions.

---

## Coming from v3 or v4

Read **[Upgrading to v6.2](UPGRADING.md)** for detailed migration notes and the full table
of retired v3 subcommands and parameters.
