<p align="center">
  <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-logo.png" alt="Vestige" width="620">
</p>

# Vestige

**Local cognitive memory for MCP agents.**

It keeps the decisions a project already made, answers lookups by exact handle, and walks a failure back through recorded evidence to the change records behind it. Every write goes through a deterministic gate, and every change or decision-informing answer lands as a signed receipt any verifier can replay. The store is the STRATA log on your machine.

[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/github/actions/workflow/status/samvallad33/vestige/ci.yml?branch=main&label=CI)](https://github.com/samvallad33/vestige/actions)
[![Binary](https://img.shields.io/badge/platforms-5_release_targets-informational)](https://github.com/samvallad33/vestige/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6)](LICENSE)

[Install](#install) · [The 4.0 surface](#the-40-surface-handles-receipts-admission) · [The Live Gate](#founding-operator) · [Continuity](#managed-continuity) · [Benchmark](#the-receipts-silent-rotation) · [Science](#the-science) · [Docs](#go-deeper)

<a id="getting-started"></a>
## The cause never looks like the bug

Agents re-learn the same lessons. They recommend a change you already tested and rejected, re-derive a fix that was already written down, and treat every session as if the last one never happened. Vestige is the local memory an MCP client calls while you work: `smart_ingest` stores, `recall` retrieves by exact handle. A restated fact reinforces the existing memory instead of duplicating it (lookalikes never merge), disagreements can be listed, unused memories fade under FSRS-6, and a failure can be walked backward through recorded evidence to its suspect change records — as labeled hypotheses, never asserted causes.

<p align="center">
  <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">
    <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/black-box-cause.gif" alt="Vestige Black Box: walking a failure back to the earlier records behind it" width="100%">
  </a>
</p>

<p align="center"><sub>A causal walk starts from explicit evidence (a failing test, a stack frame, a CI run) and follows exact mechanism edges to candidate change records. The result is a list of candidates. <code>causality_verified</code> stays false. <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">Watch the walk</a>.</sub></p>

## Install

Download a release archive from [GitHub Releases](https://github.com/samvallad33/vestige/releases/latest). No Docker, no signup, no compile step. The archives are `vestige-mcp-aarch64-apple-darwin.tar.gz` (macOS ARM), `vestige-mcp-x86_64-apple-darwin.tar.gz` (macOS Intel), `vestige-mcp-x86_64-unknown-linux-gnu.tar.gz` (Linux x86_64), `vestige-mcp-aarch64-unknown-linux-gnu.tar.gz` (Linux arm64), and `vestige-mcp-x86_64-pc-windows-msvc.zip` (Windows x86_64). Each one contains three binaries: `vestige` (the CLI), `vestige-mcp` (the MCP server), and `vestige-restore`.

Prefer Homebrew?

```bash
brew install samvallad33/tap/vestige
```

`eget samvallad33/vestige` installs those same GitHub Release archives. Do not install this version with npm. There is no model download: 4.0 carries no embedding runtime, so no ONNX Runtime or network model fetch is involved.

Connect the MCP server. The client command is `vestige-mcp`:

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

| Client | Setup |
|---|---|
| Claude Code | `claude mcp add vestige vestige-mcp -s user` |
| Codex | `codex mcp add vestige -- vestige-mcp` |
| Cursor / VS Code / Windsurf | [docs/integrations/](docs/integrations/) |
| Claude Desktop | [docs/CONFIGURATION.md](docs/CONFIGURATION.md#claude-desktop-macos) |
| Cline / Continue / Zed / Goose | the JSON above, in that client's MCP settings |

`vestige-mcp` also takes `--data-dir <PATH>`, `--http`, `--no-http`, and `--http-port <PORT>` (default 3928, and passing the flag turns HTTP on). HTTP binds to `127.0.0.1` unless `VESTIGE_HTTP_BIND` is set. `VESTIGE_AUTH_TOKEN` overrides the bearer token. `VESTIGE_HTTP_ALLOWED_ORIGINS` is a comma-separated browser allowlist. `VESTIGE_DASHBOARD_ENABLED=1` starts the dashboard from the server; `VESTIGE_DASHBOARD_PORT` defaults to 3927. `VESTIGE_SYSTEM_PROMPT_MODE` is `minimal` or `full`. `RUST_LOG` filters logs. `VESTIGE_DATA_DIR` is the data directory when `--data-dir` is absent. 4.0 never creates a SQLite store and refuses to open one: an existing v3 store must be migrated first — [docs/MIGRATING-TO-4.0.md](docs/MIGRATING-TO-4.0.md). A migrated STRATA directory lives at `<data-dir>/strata` by default.

Optional output defaults live in `<data-dir>/vestige.toml`. A missing file uses the built-in defaults. An explicit MCP argument wins over the file.

```toml
[defaults]
profile = "default"       # lean | default | audit | research
detail_level = "summary"  # brief | summary | full
limit = 10
```

`lean` presets brief detail and a limit of 5. `audit` presets full detail. `research` presets full detail and a limit of 25. `default` leaves the historical tool limits alone.

Verify the CLI: `vestige dashboard`. It binds `http://127.0.0.1:3927` (override with `--port`) and `/` redirects to `/dashboard`. There is no warm-up: no model download, no reranker, nothing to wait for. A release build compares its version with the npm registry for `vestige-mcp-server`. When a newer version is published it sends an MCP `notifications/message` on logger `vestige.update` (`newer_version_available`). It does not update itself, and a failed check is skipped. `source_sync` and `vestige sync --cloud` are the calls that use the network. Full walkthrough: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

`vestige --help` lists the CLI. The subcommands are `stats`, `health`, `consolidate`, `upgrade`, `update`, `sandwich`, `restore`, `backup`, `migrate-to-strata`, `export`, `portable-export`, `portable-import`, `sync`, `gc`, `dashboard`, `ingest`, `ingest-git`, `scan-secrets`, `backfill`, `causal-walk`, `recall`, `compose`, `project`, `selftest`, `forgotten-lesson`, and `serve`. `--data-dir` is global.

<a id="the-40-surface-handles-receipts-admission"></a>
## The 4.0 surface: handles, receipts, admission

4.0 removes resemblance from the decision path. There are no embeddings, no
FTS/keyword ranking, no hybrid or semantic search, and no similarity-based
dedup. Three mechanisms replace them.

**Lookups are handles.** `recall` accepts only exact handles — `mem:`, `sha:`,
`path:`, `line:`, `sym:`, `test:`, `run:`, `call:`, `issue:`, `pr:`, `purl:`,
`node:`, `receipt:`, `tag:` (filter only), `session:` — matched byte-exactly.
Free text is never searched: it returns the fail-closed `handle_required`
payload, with no candidates field. Full grammar:
[docs/HANDLES.md](docs/HANDLES.md).

**Writes are gate-admitted and receipted.** Every mutation runs
PROPOSE → GATE → EFFECT through a deterministic gate — the default policy
holds destructive retire actions — and lands a signed frame in the append-only
STRATA log. Ingest decisions are receipted (`ADMISSION_RECEIPT`): a fact
re-stated with identical identity (Unicode-normalized, trimmed,
whitespace-collapsed) reinforces the live node; a lookalike with no recorded
causal relation never links, merges, ranks, or conflicts. Refusals are
receipted too. Full registry: [docs/RECEIPTS.md](docs/RECEIPTS.md).

**Failure investigation walks evidence, not vibes.** `causal_walk` starts from
explicit start points (a failing test, a stack frame, a CI run, a logged
write, a version range) and follows exact mechanism edges to suspect change
records. Candidates are hypotheses, `causality_verified` stays false, and the
default preview persists nothing; explicit `promote=true` records an
`evidence_of` trail edge. `graph(action="never_composed")` surfaces memory
pairs that were never connected — a testable question, never a proven link.
`vestige migrate-to-strata --from <path>` carries a v3 SQLite store into the
STRATA log read-only — the original file is never modified and stays as your
backup. Guide: [docs/MIGRATING-TO-4.0.md](docs/MIGRATING-TO-4.0.md).

A memory whose validity window has closed is superseded, not deleted:
supersede and correct are appends, and only an explicit
`memory(action="purge", confirm=true)` removes content (a content-free
tombstone stays for audit).

<a id="founding-operator"></a>
## 🛡️ Founding Operator

Review of writes, on the machine. The default mode is `fast`: a write auto-commits. With tracing on, a retrieval still leaves a receipt. The mode is `<data-dir>/review_mode.json`, written by `POST /api/memory-prs/mode`. A missing or unreadable file is `fast`.

| Mode | What a write does |
|---|---|
| `fast` | Auto-commit. Nothing is held for approval. |
| `risk_gated` | Ordinary writes auto-commit. A risky write opens a Memory PR: it contradicts a high-trust memory, supersedes, merges, forgets, or touches identity, preference, auth, money, or a similar topic. A live node can be held out of retrieval until the PR is decided. |
| `paranoid` | Every write opens a Memory PR. |

With tracing on, the calling agent sees `memoryPrs` on the tool result when a PR opens. Review them in the dashboard under Memory PRs, or `GET /api/memory-prs`. In `fast` mode a destructive write is not gated. In `risk_gated` and `paranoid`, that write is recorded on a PR after it has already been applied: there is no live node to hold.

**Recorded run: [THE LIVE GATE](https://github.com/samvallad33/vestige/releases/tag/launch-night-live-gate-20260914).**

<a id="managed-continuity"></a>
<a id="vestige-pro"></a>
## 🔄 Managed Continuity

The local store moves with a backup, a portable archive, or hosted sync. `vestige backup` copies the store file, so the copy has every table. `vestige sync` moves the portable archive: memories, FSRS state, the graph, intentions, and the other tables in that format. Receipts stay in the STRATA log.

```bash
# SQLite backup of the local store
vestige backup ~/vestige-backup.db

# Two-way sync with a portable archive you keep yourself
vestige sync ~/vestige-archive.vportable

# Restore that archive into another data directory
vestige --data-dir ~/new-machine-store sync ~/vestige-archive.vportable
```

File sync writes the portable archive as-is. Hosted sync is the encrypted path. `vestige sync --cloud` requires `VESTIGE_CLOUD_ENDPOINT` (or `--endpoint`), `VESTIGE_CLOUD_SYNC_KEY`, and `VESTIGE_CLOUD_ENCRYPTION_KEY`. The archive is encrypted on the device with XChaCha20-Poly1305 before upload. The key is Argon2id over a passphrase of at least 12 characters. The passphrase never leaves the process. The server stores ciphertext. Lose the passphrase and the archive cannot be decrypted.

The CLI prices hosted sync at $19/month and points here: [Vestige Pro](https://github.com/samvallad33/vestige#vestige-pro).

## The receipts: Silent Rotation

| Arm (6 models, 25 trials) | Converged correct | Converged wrong | Split |
|---|---|---|---|
| No memory | 0/25 | **21/25** | 4/25 |
| Dense cosine RAG | 4/23 | **12/23** | 7/23 |
| Vestige | 20/23 | **0/23** | 3/23 |

Outcomes come from `tests/by_model_tables.py`: a trial is correct when the tests were green, the production replay passed, and the key was right; wrong when the tests were green but production failed; split when the merge conflicted.

## The science

The mechanisms below are implemented in the 4.0 engine. Write-up: [docs/SCIENCE.md](docs/SCIENCE.md).

| Mechanism | What it does | Source in the code |
|---|---|---|
| Prediction-error gating | Stores the novel; an identical restatement reinforces instead of duplicating | `smart_ingest` |
| Exact-handle recall | Byte-exact handles, no resemblance anywhere in retrieval | `recall`, [docs/HANDLES.md](docs/HANDLES.md) |
| Gate-admitted writes | PROPOSE → GATE → EFFECT; destructive actions hold by default | `strata-gate`, `strata-store` |
| FSRS-6 | 21-parameter retrievability. Used memories persist, unused ones fade | `fsrs`, `strata-kernel` |
| Signed causal log | Every change is a frame in an append-only, verifiable chain | `strata`, [docs/RECEIPTS.md](docs/RECEIPTS.md) |
| Causal walk | Walks explicit evidence to candidate change records; hypotheses, not causes | `causal_walk` |
| Never-composed pairs | Live memory pairs with no recorded relation — testable fusion candidates | `graph(action="never_composed")` |
| Dual-strength | Storage strength and retrieval strength, tracked separately | Bjork & Bjork 1992 |
| Memory dreaming | Replay and synthesis during consolidation | `dreams`, reached through `maintain` action `dream` |
| Active forgetting | Top-down suppression, reversible, distinct from FSRS decay | Anderson 2025, Davis 2020, in `active_forgetting` |

## The tools

`tools/list` advertises these 18 tools, sorted by name. The list is compact: discriminator enums stay on the wire, and deep fields move one call deeper. The serialized catalog stays under 20 KiB. `memory_status` with `view=tools` lists every tool; set `tool` to a name for that tool's full input schema. Hidden aliases from older releases still dispatch, and they are not in the list.

| Tool | Purpose |
|---|---|
| `causal_walk` | Investigate a failure from explicit start points (failing test, stack frame, CI run, logged write, version range) through exact mechanism edges to suspect change records. Hypotheses, not proven causes; preview persists nothing |
| `codebase` | Remember a pattern or decision, fetch context, verify anchors, reanchor evidence |
| `dedup` | Exact-identity scan, merge/supersede plans, apply, undo, tag maintenance. Merged memories are invalidated, never deleted |
| `forgotten_lesson` | Decayed fix/lesson memories sharing an exact anchor with a failure |
| `graph` | Chains, associations, bridges, predictions, composition topology including `never_composed` pairs. `label` is the write |
| `intention` | Set, check, update, list. `graph` runs the evidence-aware plan |
| `maintain` | `consolidate`, `dream`, `gc` (dry run by default), `importance_score`, `backup`, `export`, `restore` |
| `memory` | `get`, `get_batch`, `state`, `promote`, `demote`, `edit`, `purge` (`confirm=true`). `delete` aliases `purge`. Demote does not delete |
| `memory_status` | `health`, `retention`, `timeline`, `changelog`, `stats`, `tools`. `view=tools` plus `tool` unfolds one full schema |
| `project` | Preview a fenced region of `CLAUDE.md` or `MEMORY.md`. `write` needs `confirm=true` and replaces only the fence |
| `purge` | Remove one memory's content for good. Irreversible; `confirm=true` required. `destructiveHint` is true, and `_meta["anthropic/requiresUserInteraction"]` is true, so the client prompts. Same path as `memory` action `purge`. The tombstone is an opaque marker; the reason is logged, not stored |
| `receipt` | `get` a stored receipt, or `replay` it with named slots withheld (no rerun, no model, no causal claim) |
| `recall` | Look up by exact handle. Returns the causal neighborhood with a signed RECALL receipt; free text returns `handle_required` |
| `selftest` | Planted-cause selftest on a scratch copy: does the walk find the planted cause? |
| `session_start` | Memories, open intentions, status, predictions, and codebase context under one budget |
| `smart_ingest` | Save through prediction-error gating: created, reinforced, or superseded. `items` batches up to 20 |
| `source_sync` | Index GitHub (`GITHUB_TOKEN`) or Redmine (`REDMINE_URL`, `REDMINE_API_KEY`) into local memories |
| `suppress` | Hold a memory out of retrieval and speed its decay. `reverse=true` undoes it inside 24 hours when the stored snapshot still matches |

Full contracts: [docs/TOOL-CONTRACTS.md](docs/TOOL-CONTRACTS.md) · Handle grammar: [docs/HANDLES.md](docs/HANDLES.md) · Hygiene and standing habits: [docs/MEMORY_HYGIENE.md](docs/MEMORY_HYGIENE.md)

## The dashboard

```bash
vestige dashboard
```

The server binds **http://127.0.0.1:3927** and redirects `/` to **/dashboard**. The observatory steps a fixed 60fps clock, 720 frames, 12 seconds, and can export that loop as an mp4. Share artifacts are structure-only: the shape of the store, not the memory text.

## Under the hood

| | |
|---|---|
| Engine | Rust 2024. Release archives ship `vestige`, `vestige-mcp`, and `vestige-restore` |
| Retrieval | Exact handles only (`mem:`, `sha:`, `path:`, `line:`, `sym:`, `test:`, `run:`, `call:`, `issue:`, `pr:`, `purl:`, `node:`, `receipt:`, `tag:`, `session:`). No embeddings, no FTS, no reranker; free text returns `handle_required` |
| Storage | The STRATA signed append-only log (`strata-store`). v3 SQLite is a read-only migration source: opening one is refused, and `vestige migrate-to-strata` carries it over without modifying it. See [docs/STORAGE.md](docs/STORAGE.md) and [docs/MIGRATING-TO-4.0.md](docs/MIGRATING-TO-4.0.md) |
| First run | No model downloads and no warm-up. A release build hints when npm has a newer `vestige-mcp-server`, and does not update itself |

## Go deeper

[Getting Started](docs/GETTING-STARTED.md) · [Handles](docs/HANDLES.md) · [Receipts](docs/RECEIPTS.md) · [Migrating to 4.0](docs/MIGRATING-TO-4.0.md) · [FAQ](docs/FAQ.md) · [The Science](docs/SCIENCE.md) · [Configuration](docs/CONFIGURATION.md) · [Storage](docs/STORAGE.md) · [Tool contracts](docs/TOOL-CONTRACTS.md) · [Changelog](CHANGELOG.md)

## License

AGPL-3.0-only. Hosted Continuity (`vestige sync --cloud`) is a separate service.
