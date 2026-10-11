# CLI → MCP audit (Vestige 4.2.0 default branch)

Sam’s direction: **agents drive Vestige through MCP tools**, not shell recipes. This document inventories every shipped CLI surface, maps it to the sixteen advertised MCP tools (see [`TOOL-CONTRACTS.md`](TOOL-CONTRACTS.md)), and proposes `gather` as a new MCP-only workflow.

**Scope of binaries**

| Binary | Role |
|--------|------|
| `vestige` | Primary CLI (`crates/vestige-mcp/src/bin/cli.rs`) |
| `vestige-mcp` | MCP server (`crates/vestige-mcp/src/main.rs`) |
| `vestige-upgrade` | One-shot v3 SQLite → Strata import (`crates/vestige-upgrade/src/main.rs`) |
| `vestige-restore` | JSON backup re-ingest helper (`crates/vestige-mcp/src/bin/restore.rs`) |
| `strata-verify` | Standalone log verifier (`crates/strata-verify/src/bin/strata_verify.rs`) |
| `make-v3-fixture` | Dev-only migration fixture generator (`crates/strata-migrate/src/bin/make-v3-fixture.rs`) |

Test-only binaries (`tests/differential`, etc.) are out of scope for product CLI.

**MCP tool registry (advertised on Strata 4.2.0)**

`recall`, `smart_ingest`, `memory`, `memory_status`, `receipt`, `session_start`, `causal_walk`, `forgotten_lesson`, `selftest`, `ghostlink`, `codebase`, `intention`, `maintain`, `dedup`, `project`, `suppress`. Deprecated aliases still dispatch: `ingest`, `session_checkpoint`. `purge` and `source_sync` appear in code but are withheld or feature-gated on default release builds.

---

## Summary table

| Command | Defined in | MCP equivalent | Verdict | Gap |
|---------|------------|----------------|---------|-----|
| `vestige` (global `--data-dir`) | `cli.rs` `Cli` | env `VESTIGE_DATA_DIR` / server `--data-dir` | KEEP CLI-ONLY | MCP session has no per-call data-dir override once bound |
| `vestige stats` | `cli.rs` `Commands::Stats` | `memory_status` `view=stats` | REPLACE | — |
| `vestige health` | `Commands::Health` | `memory_status` `view=health` | REPLACE | — |
| `vestige consolidate` | `Commands::Consolidate` | `maintain` `action=consolidate` (withheld on Strata) | REPLACE | Strata: no-op / `unavailable_in_4_0` |
| `vestige upgrade` | `Commands::Upgrade` → `run_upgrade` | — | KEEP CLI-ONLY | Spawns `vestige-upgrade`; needs store lock before MCP serves |
| `vestige update` | `Commands::Update` → `run_update` | — | KEEP CLI-ONLY | Downloads GitHub release artifacts; installs binaries |
| `vestige sandwich install` | `SandwichCommands::Install` | — | KEEP CLI-ONLY | Writes hook/agent files under user home; optional network |
| `vestige restore <file>` | `Commands::Restore` | `maintain` `restore` (withheld); batch `smart_ingest` | FOLD IN | `maintain.restore` + `confirm`; id mapping unlike raw JSON round-trip |
| `vestige backup <dir>` | `Commands::Backup` | `maintain` `action=backup` | REPLACE | MCP writes under `<data-dir>/backups/` not arbitrary path |
| `vestige migrate-to-strata` | `Commands::MigrateToStrata` | — | KEEP CLI-ONLY | Offline migration; uses `strata-migrate` / read-only v3 |
| `vestige strata-verify <dir>` | `Commands::StrataVerify` | — | FOLD IN | Proposed `maintain` `action=strata_verify` or `memory_status` diagnostic view |
| `vestige export` | `Commands::Export` | `maintain` `action=export` | REPLACE | MCP path under `exports/`; filters via args |
| `vestige portable-export` | `Commands::PortableExport` | `maintain` `export` `format=portable` (withheld) | KEEP CLI-ONLY | Legacy SQLite only; Strata refuses |
| `vestige portable-import` | `Commands::PortableImport` | — | KEEP CLI-ONLY | Legacy SQLite only |
| `vestige sync` | `Commands::Sync` | `source_sync` (withheld w/o `connectors`) | KEEP CLI-ONLY | Legacy SQLite + cloud sync; not on Strata 4.x default |
| `vestige gc` | `Commands::Gc` | `maintain` `action=gc` (stub) | FOLD IN | MCP stub; CLI `--dry-run` lists candidates — expose as `maintain.gc` `dry_run=true` |
| `vestige dashboard` | `Commands::Dashboard` | HTTP `vestige serve` / attach | KEEP CLI-ONLY | Starts UI; competes with `.serve.lock` unless attach |
| `vestige ingest` | `Commands::Ingest` | `smart_ingest` | REPLACE | CLI `ago_days` / `created_at` refused on Strata |
| `vestige ingest-git` | `Commands::IngestGit` | `codebase` `ingest_repo` (partial) | FOLD IN | Strata refuses CLI; MCP `ingest_repo` is commit-scoped with paging |
| `vestige connect` | `Commands::Connect` → `run_connect` | — (auto-connect at ingest) | FOLD IN | Proposed `maintain` `action=connect` `{scope,dryRun,maxEdges}` |
| `vestige scan-secrets` | `Commands::ScanSecrets` | `memory_status` `stats` hygiene (partial) | FOLD IN | Proposed `maintain` `action=scan_secrets` with `--json` parity |
| `vestige backfill` | `Commands::Backfill` | `causal_walk` (Strata) | REPLACE | Legacy SQLite + BM25 `contrast`; Strata refuses |
| `vestige causal-walk` | `Commands::CausalWalk` | `causal_walk` | REPLACE | CLI holds store lock; MCP needs exclusive writer or attach |
| `vestige recall` | `Commands::Recall` | `recall` | REPLACE | Free-text `query` legacy only |
| `vestige compose` | `Commands::Compose` → `run_compose` | `ghostlink` `mode=propose` | REPLACE | CLI blocks if another writer holds log |
| `vestige project` | `Commands::Project` | `project` `preview` / `write` | REPLACE | CLI writes local files; MCP needs `path` + `confirm` |
| `vestige serve` | `Commands::Serve` | `vestige-mcp --http` | KEEP CLI-ONLY | Starts HTTP MCP + optional dashboard |
| `vestige selftest` | `Commands::Selftest` | `selftest` | REPLACE | — |
| `vestige forgotten-lesson` | `Commands::ForgottenLesson` | `forgotten_lesson` | REPLACE | — |
| `vestige prove` | `walk_verify::ProveArgs` in `cli.rs` | — | FOLD IN | Long-running git + test oracle; needs store writer + subprocess model |
| `vestige _prove` (hidden) | `Commands::ProveChild` | — | KEEP CLI-ONLY | `git bisect run` child; not agent-facing |
| `vestige-mcp` | `main.rs` `parse_args` | (is the server) | KEEP CLI-ONLY | stdio default; HTTP via `--http` / env |
| `vestige-upgrade` | `vestige-upgrade/src/main.rs` | — | KEEP CLI-ONLY | v3 import + `.serve.lock` |
| `vestige-restore` | `restore.rs` `main` | batch `smart_ingest` | FOLD IN | Fold into `maintain.restore` or document MCP-only path |
| `strata-verify` | `strata_verify.rs` | — | FOLD IN | Same as `vestige strata-verify` |
| `make-v3-fixture` | `make-v3-fixture.rs` | — | KEEP CLI-ONLY | Maintainer/test fixture only |

---

## Per-command detail

### Global

- **`vestige --data-dir DIR`** — `Cli` in `cli.rs`. Sets `CLI_DATA_DIR` for one process. **MCP:** server uses `VESTIGE_DATA_DIR` or platform default at startup. **Verdict:** KEEP CLI-ONLY for bootstrap. **Gap:** no per-tool data-dir on an already-running server.

### Store introspection

- **`stats`** (`--tagging`, `--states`) — `run_stats`. Prints counts / distributions. **MCP:** `memory_status` `view=stats` (hygiene aggregates in 4.2.0). **REPLACE.**

- **`health`** — `run_health`. **MCP:** `memory_status` `view=health`. **REPLACE.**

- **`consolidate`** — `run_consolidate`. Legacy SQLite runs decay passes; Strata no-op. **MCP:** `maintain` `consolidate` → `unavailable_in_4_0` on Strata. **REPLACE** (with honest refusal).

### Install / migration / packaging

- **`upgrade`** (`--dry-run`) — shells to `vestige-upgrade` or reports. **KEEP CLI-ONLY:** must run before or outside MCP owner lock.

- **`update`** (`--version`, `--install-dir`, `--dry-run`, `--no-sandwich`, `--sandwich-companion`, sandwich flatten opts) — `run_update`. **KEEP CLI-ONLY:** binary install from GitHub.

- **`sandwich install`** (`--version`, `SandwichInstallOptions`: `--force`, `--enable-preflight`, `--enable-sandwich`, `--enable-sanhedrin`, `--with-launchd`, `--include-memory-loader`, `--sanhedrin-endpoint`, `--sanhedrin-model`, hidden `--src`) — `run_sandwich_install`. **KEEP CLI-ONLY:** local IDE hook staging.

- **`migrate-to-strata`** (`--from`, `--to`, `--dry-run`, `--accept-wal-snapshot`) — `run_migrate_to_strata`. **KEEP CLI-ONLY:** one-time migration tooling.

- **`strata-verify`** (`dir`, `--expect-key`) — `run_strata_verify` / `strata_verify::verify_path`. Read-only chain check. **FOLD IN:** `maintain` `action=strata_verify` `{dir, expectKey?}` returning JSON report.

### Backup / export / legacy sync

- **`backup`** — `run_backup`. Seals Strata `log/`. **MCP:** `maintain` `backup`. **REPLACE** (path differs).

- **`restore`** — `run_restore`. Re-ingests export JSON as new records. **FOLD IN:** `maintain` `restore` `{path, scope?}` with `confirm=true` (currently withheld).

- **`export`** (`output`, `--format`, `--tags`, `--since`) — `run_export`. **MCP:** `maintain` `export`. **REPLACE.**

- **`portable-export` / `portable-import` / `sync`** — legacy SQLite portable archive and cloud sync. **KEEP CLI-ONLY** on Strata default build.

- **`gc`** (`--min-retention`, `--max-age-days`, `--dry-run`, `--yes`) — `run_gc`. Strata delete withheld; dry-run lists. **FOLD IN:** extend `maintain.gc` with `dryRun`, `minRetention`, `maxAgeDays`.

### Operator servers

- **`dashboard`** (`--port`, `--no-open`) — `run_dashboard`. **KEEP CLI-ONLY** or attach to running `vestige-mcp` owner.

- **`serve`** (`--port`, `--dashboard`, `--dashboard-port`) — `run_serve`. **KEEP CLI-ONLY:** starts HTTP MCP (`vestige-mcp --http` equivalent).

### Memory writes & graph

- **`ingest`** — same gate as `smart_ingest` on Strata. **REPLACE** → `smart_ingest`.

- **`ingest-git`** — legacy source upsert by sha. **FOLD IN** → `codebase.ingest_repo` with `repoPath`, `since`/`until` via `rev`, `limit`, `dryRun`.

- **`connect`** — `vestige_mcp::auto_connect` full-scope scan. **FOLD IN** → `maintain.connect` reusing `auto_connect::scan_scope` + edge writes.

- **`scan-secrets`** — `strata_memory` full-store scan via `scan_secrets`. **FOLD IN** → dedicated action (hygiene stats do not expose full credential audit).

### Investigation

- **`backfill`** — legacy entity overlap + optional BM25 `contrast`. Strata refuses. **REPLACE** → `causal_walk` with explicit `start_points`.

- **`causal-walk`** — calls `tools/causal_walk` paths from CLI. **REPLACE** → `causal_walk` tool (same JSON with `--json`).

- **`recall`** — Strata: `--handle` only. **REPLACE** → `recall`.

- **`compose`** — Strata: `ghostlink` propose lenses. **REPLACE** → `ghostlink` `mode=propose`.

- **`forgotten-lesson`** — **REPLACE** → `forgotten_lesson`.

- **`selftest`** — **REPLACE** → `selftest`.

- **`prove`** — `walk_verify::prove` / `check`. Walk + git worktree tests + bisect. **FOLD IN** → MCP tool `prove` with same `ProveArgs` schema (async job + progress parts). **`prove --check`** can stay read-only without store.

- **`_prove`** (hidden) — bisect child. **KEEP CLI-ONLY.**

### Projection

- **`project`** — **REPLACE** → `project` `preview` / `write`.

### `vestige-mcp`

Flags: `--data-dir`, `--http`, `--no-http`, `--http-port`, `--help`, `--version`; env: `VESTIGE_DATA_DIR`, `VESTIGE_HTTP_*`, `VESTIGE_DASHBOARD_*`, `RUST_LOG`, `VESTIGE_AUTH_TOKEN`. Attaches via `attach::elect` when another owner holds the log. **KEEP CLI-ONLY** (bootstrap MCP).

### `vestige-upgrade`

Only `--data-dir` (+ help/version). **KEEP CLI-ONLY.**

### `vestige-restore`

`<backup.json>`. **FOLD IN** to `maintain.restore`.

### `strata-verify`

Same as `vestige strata-verify`. **FOLD IN.**

### `make-v3-fixture`

Writes synthetic v3 SQLite for tests. **KEEP CLI-ONLY.**

---

## Cross-cutting gaps (MCP-first blockers)

1. **Single writer / `.serve.lock`** — CLI commands that call `open_storage()` take the writer lock. A running `vestige-mcp` owner blocks `compose`, `connect`, `prove`, and `causal-walk` CLI. MCP clients should use tools on the owner; attach mode proxies stdio but does not remove lock contention for second writers.

2. **Scope mismatch** — CLI defaults `scope=user`; `codebase.ingest_repo` defaults scope to codebase name. Agents must pass explicit `scope` on every write and walk or handles land in the wrong namespace.

3. **`ingest_repo` pagination** — 45s budget and `pageBackWith`; a `gather` or large repo sync must loop with `rev` cursor, not one shot.

4. **HTTP MCP headers** — Streamable HTTP clients must send `MCP-Protocol-Version` and session ids per spec; misconfigured clients see opaque 4xx.

5. **`prove` subprocess model** — Bisect children cannot write Strata; only the parent ingests. An MCP `prove` needs job streaming, not a single synchronous tool result.

6. **`source_sync` vs product law** — Feature-gated connector indexes issues with semantic recall in its docstring; default 4.x withholds it. `gather` must not depend on embeddings or keyword search.

---

## New MCP tool: `gather`

**Not a CLI command.** Orchestrates receipt-backed ingestion from a single GitHub issue URL, repo commit coverage, then deterministic `causal_walk`.

### Law

- **Gathering is dynamic** (network fetch, git paging).
- **Judging is deterministic** (`causal_walk`, `prove`, `receipt` replay).
- Records only **exact identifiers**: SHAs, `owner/repo#N`, file:line anchors, CI run URLs/IDs, version tags, package versions.
- No BM25, embeddings, name fuzzy match, or SQLite in main binaries.

### Input schema

```json
{
  "type": "object",
  "required": ["issueUrl", "scope"],
  "properties": {
    "issueUrl": {
      "type": "string",
      "description": "Canonical https://github.com/owner/repo/issues/N"
    },
    "scope": {
      "type": "string",
      "description": "Strata scope for all writes in this gather (required; no default)"
    },
    "repoPath": {
      "type": "string",
      "description": "Local checkout top for ingest_repo and blame anchors"
    },
    "workedIn": { "type": "string", "description": "Optional last-good tag/ref from issue text (exact)" },
    "brokeIn": { "type": "string", "description": "Optional first-bad tag/ref from issue text (exact)" },
    "symptomMemoryId": {
      "type": "string",
      "description": "If already ingested, walk from this node; else created from issue receipt"
    },
    "ingestRepo": {
      "type": "object",
      "properties": {
        "limit": { "type": "integer", "default": 100, "maximum": 500 },
        "dryRun": { "type": "boolean", "default": false }
      }
    },
    "walk": {
      "type": "object",
      "description": "Passed to causal_walk after gather completes",
      "properties": {
        "startPoints": { "type": "array" },
        "nodeId": { "type": "string" }
      }
    }
  }
}
```

### Output schema

```json
{
  "gatherId": "eff-…",
  "issue": { "owner", "repo", "number", "url" },
  "receipts": [
    { "label": "issue_body|comment|pr|commit|ci_run|ingest_repo_page", "receiptId": "eff-…", "handles": ["owner/repo#N", "commit:sha", …] }
  ],
  "ingestRepo": {
    "commitsWritten": 0,
    "remaining": 0,
    "pageBackWith": "sha~1",
    "gitStoppedEarly": null
  },
  "causalWalk": { /* full causal_walk tool response */ },
  "incomplete": ["ingest_repo_remaining", "github_rate_limit"] 
}
```

### Flow

1. Parse `issueUrl` → `owner`, `repo`, `number` (no search).
2. Fetch issue + comments + linked PRs + commits (GitHub REST; token from env only, never args) — each response stored via `smart_ingest` with tags `github-issue`, `owner/repo#N`, `commit:<sha>`, `ci-run:<id>` as present in payloads.
3. Loop `codebase` `ingest_repo` until `workedIn..brokeIn` window covered (or `remaining=0`), using `pageBackWith`.
4. Emit labeled receipts (`receipt` `get` compatible) for each artifact class.
5. Call `causal_walk::execute` with `start_points` including `logged_write` / `version_range` / `ci_run` derived only from recorded handles.

### Reuse

| Step | Existing code |
|------|----------------|
| Issue/PR HTTP | Patterns from `tools/source_sync.rs` (split per-issue, no semantic index) |
| Commits | `tools/repo_ingest.rs` / `codebase` `ingest_repo` |
| Receipts | Strata gate + `tools/receipt.rs` |
| Walk | `tools/causal_walk.rs` |
| Entity extraction | `smart_ingest` scanners (`CommitSha`, `IssueRef`, `Url`, `Version`) |
| Idempotent writes | `smart_ingest` `intent_id` per upstream id |

---

## Build plan (ordered)

1. **Document & deprecate** — Point README/integrations at MCP; mark CLI investigation commands as thin wrappers in help text (no behavior change yet).
2. **`maintain` expansions** — `restore`, `strata_verify`, `scan_secrets`, `connect`, real `gc` dry-run JSON (admission sweep updates).
3. **`gather` tool** — New module orchestrating GitHub fetch + `ingest_repo` loop + `causal_walk`; gate behind optional `connectors` or dedicated `gather` feature; stdio admission test.
4. **`prove` MCP job** — Async tool with progress notifications; keep `_prove` CLI child.
5. **Lock / attach story** — Document attach for dashboard; HTTP single-writer; optional read-only causal_walk attach path (future).
6. **Remove or hide** — `backfill`, `ingest-git`, `sync` from default help once MCP parity shipped; keep `legacy-sqlite` feature binaries for harnesses only.

---

## Counts

| Verdict | Count |
|---------|------:|
| REPLACE | 14 |
| FOLD IN | 11 |
| KEEP CLI-ONLY | 15 |
| **Total CLI surfaces audited** | **40** |

(30 `vestige` subcommands including hidden `_prove`, plus 4 sibling binaries, plus `vestige-mcp` server entry, plus `sandwich` counted under `sandwich install`, global flag documented separately.)

*Generated by CLI→MCP audit pass; code references Vestige 4.2.0 on `main`.*
