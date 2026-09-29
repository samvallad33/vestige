<p align="center">
  <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-logo.png" alt="Vestige" width="620">
</p>

# Vestige

**Local cognitive memory for MCP agents.**

It keeps the decisions a project already made, and it can reach backward from a failure to earlier memories that share its entities. With tracing on, a retrieval leaves a receipt. The store is SQLite on your machine.

[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/github/actions/workflow/status/samvallad33/vestige/ci.yml?branch=main&label=CI)](https://github.com/samvallad33/vestige/actions)
[![Binary](https://img.shields.io/badge/platforms-5_release_targets-informational)](https://github.com/samvallad33/vestige/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6)](LICENSE)

[Install](#install) · [Why not RAG](#why-not-just-rag) · [The Live Gate](#founding-operator) · [Continuity](#managed-continuity) · [Benchmark](#the-receipts-silent-rotation) · [Science](#the-science) · [Docs](#go-deeper)

<a id="getting-started"></a>
## The cause never looks like the bug

Agents re-learn the same lessons. They recommend a change you already tested and rejected, re-derive a fix that was already written down, and treat every session as if the last one never happened. Vestige is the local memory an MCP client calls while you work: `smart_ingest` stores, `recall` retrieves. Near-identical writes merge, disagreements can be listed, unused memories fade under FSRS-6, and a failure can be walked backward to earlier memories that share its entities.

<p align="center">
  <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">
    <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/black-box-cause.gif" alt="Vestige Black Box: resemblance search beside a backward backfill from a failure" width="100%">
  </a>
</p>

<p align="center"><sub><code>vestige backfill --contrast</code> prints what resemblance ranks first, then the earlier memories that share entities with the failure. The result is a list of candidates. <code>causality_verified</code> stays false. <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">Watch the walk</a>.</sub></p>

## Install

Download a release archive from [GitHub Releases](https://github.com/samvallad33/vestige/releases/latest). No Docker, no signup, no compile step. The archives are `vestige-mcp-aarch64-apple-darwin.tar.gz` (macOS ARM), `vestige-mcp-x86_64-apple-darwin.tar.gz` (macOS Intel), `vestige-mcp-x86_64-unknown-linux-gnu.tar.gz` (Linux x86_64), `vestige-mcp-aarch64-unknown-linux-gnu.tar.gz` (Linux arm64), and `vestige-mcp-x86_64-pc-windows-msvc.zip` (Windows x86_64). Each one contains four binaries: `vestige` (the CLI), `vestige-mcp` (the MCP server), `vestige-restore`, and `vestige-upgrade` (the v3 importer).

Prefer Homebrew?

```bash
brew install samvallad33/tap/vestige
```

`eget samvallad33/vestige` installs those same GitHub Release archives. Do not install this version with npm. Intel Mac builds are linked against a system ONNX Runtime: `brew install onnxruntime` before the first run, and set `ORT_DYLIB_PATH` to that library. That tarball includes `INSTALL-INTEL-MAC.md`.

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

`vestige-mcp` also takes `--data-dir <PATH>`, `--http`, `--no-http`, and `--http-port <PORT>` (default 3928, and passing the flag turns HTTP on). HTTP binds to `127.0.0.1` unless `VESTIGE_HTTP_BIND` is set. `VESTIGE_AUTH_TOKEN` overrides the bearer token. `VESTIGE_HTTP_ALLOWED_ORIGINS` is a comma-separated browser allowlist. `VESTIGE_DASHBOARD_ENABLED=1` starts the dashboard from the server; `VESTIGE_DASHBOARD_PORT` defaults to 3927. `VESTIGE_SYSTEM_PROMPT_MODE` is `minimal` or `full`. `RUST_LOG` filters logs. `VESTIGE_DATA_DIR` is the data directory when `--data-dir` is absent. The database file inside that directory is always `vestige.db`.

Optional output defaults live in `<data-dir>/vestige.toml`. A missing file uses the built-in defaults. An explicit MCP argument wins over the file.

```toml
[defaults]
profile = "default"       # lean | default | audit | research
detail_level = "summary"  # brief | summary | full
limit = 10
```

`lean` presets brief detail and a limit of 5. `audit` presets full detail. `research` presets full detail and a limit of 25. `default` leaves the historical tool limits alone.

Verify the CLI: `vestige dashboard`. It binds `http://127.0.0.1:3927` (override with `--port`) and `/` redirects to `/dashboard`. The first start of `vestige-mcp` downloads the Nomic embedding model (about 130 MB) and logs the milestone that keyword search works immediately and semantic ranking joins when the runtime is ready. Until then, `recall` and a save that stored no vector carry a `warming` block. The Jina reranker (about 150 MB) loads in the background; until it does, ranking stays BM25. After those downloads, memory calls stay on the machine. `source_sync` (`--features connectors`) and `vestige sync --cloud` (`--features cloud-sync`) are the calls that use the network. Neither feature is on in a 4.0 default build. Full walkthrough: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

`vestige --help` lists the CLI. The subcommands are `stats`, `health`, `consolidate`, `upgrade`, `update`, `sandwich`, `embeddings`, `restore`, `backup`, `export`, `portable-export`, `portable-import`, `sync`, `gc`, `dashboard`, `ingest`, `scan-secrets`, `backfill`, `recall`, `compose`, `project`, and `serve`. `--data-dir` is global.

## Why not just RAG?

`recall` with `mode=lookup` is hybrid keyword and semantic search. That ranks text that resembles the query. Backfill ranks a different set: earlier memories that share entities with a recorded failure.

| | Resemblance search | Vestige backfill |
|---|---|---|
| Retrieval basis | Similarity to the query | Shared entities and age, inside a lookback window |
| What `--contrast` prints | The top lookalikes for the failure text | Then the backward candidates |
| Contradictions | Both stored, both returned by lookup | `recall` mode `contradictions` lists disagreement pairs for a topic |
| Redundant writes | Accumulate | `smart_ingest` merges near-identical content through prediction-error gating |
| Unused memories | Persist at full weight | Fade under FSRS-6 |
| Your data | Often a hosted index | SQLite in the data directory. Cloud sync is a separate command |

The backward reach is the Retroactive Salience Backfill port of Zaki, Cai et al., *Nature* 2024 (637:145-155): a salient failure can surface an earlier memory that shares entities with it. The implementation is backward-only. The receipt claim boundary is explicit-entity candidate evidence, not an asserted cause. Each candidate's reason says shared entities support an association, not proof of cause. The CLI prints those hits as associated candidates through shared entities, and a similarity rank in that scan means the candidate was quiet then. The MCP headline asks you to investigate before attributing cause. `causality_verified` stays false.

`vestige backfill --contrast` prints the resemblance ranking, then the candidates. The CLI promotes those candidates unless you pass `--no-promote`. The MCP `backfill` tool does the opposite: `promote` defaults to false, so a preview writes no edge and changes no strength. `promote=true` records a `backfill_candidate` edge and reinforces eligible memories. `failure_id` defaults to the latest failure-like memory in `scope` (default `user`). `lookback_days` defaults to 30. With tracing on (the default; `VESTIGE_TRACE=0` turns it off), a triggered MCP backfill that returns candidates saves a receipt. `path_ids` is the candidate-to-failure route when an edge was stored. A path shorter than two ids does not render.

`recall` rewrites a short query before fusion. Snake_case, CamelCase, and kebab-case split into words, and a query of two to five words also gets the wrapper `notes about … for the current task`. The original string is always the first pass. Pass *n* (the original is pass 0) multiplies the score by 0.95 to that power, so a variant only rescues a memory the original missed. A query that is already prose, with nothing to split, runs once. No model is called.

Below `abstain_floor` (default 0.35) a weak match is not an answer. `recall` sets `abstained` true, returns an empty `results` list, and puts the nearest matches on `nearest`. A floor of 1 always answers. An empty match list is not an abstention.

A memory whose validity window has closed is withheld from current results, counted in `supersededWithheld`. `include_superseded=true` keeps those memories and multiplies a closed window's score by 0.1. A `validAt` query keeps history at its original score. `source_status=tombstoned` is the other request that asks for non-current records.

On `detail_level=full`, precision is retention squared. Under 0.45 the hit is a gist of about 200 characters with `precisionLow`, not the full text. `memory` action `get` still returns the content. The default `summary` detail still returns content.

A retrieval that returns two or more memories writes a `narrative` edge on at most the top three rank-ordered pairs. Strength is 0.2 times how often that pair comes back together, capped at 0.6.

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

The local store moves with a SQLite backup, a portable archive, or hosted sync. `vestige backup` is `VACUUM INTO`, so the copy has every table. `vestige sync` moves the portable archive: memories, embeddings, FSRS state, the graph, intentions, and the other tables in that format. Receipts and Memory PRs stay in the SQLite file.

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

The mechanisms below are implemented in the Rust engine. Write-up: [docs/SCIENCE.md](docs/SCIENCE.md).

| Mechanism | What it does | Source in the code |
|---|---|---|
| Prediction-error gating | Stores the novel; merges near-identical content | `prediction_error` |
| FSRS-6 | 21-parameter retrievability. Used memories persist, unused ones fade | `fsrs` |
| Retroactive Salience Backfill | Reaches backward from a failure along shared entities | Zaki, Cai et al. 2024, *Nature*, in `retroactive_backfill` |
| Synaptic tagging | Marks a memory so a later salient event can consolidate it | Redondo & Morris 2011, in `synaptic_tagging` |
| Spreading activation | One retrieval activates related memories | Collins & Loftus 1975 |
| Dual-strength | Storage strength and retrieval strength, tracked separately | Bjork & Bjork 1992 |
| Memory dreaming | Replay and synthesis during consolidation | `dreams`, reached through `maintain` action `dream` |
| Active forgetting | Top-down suppression, reversible, distinct from FSRS decay | Anderson 2025, Davis 2020, in `active_forgetting` |
| Narrative edges | Co-retrieved memories gain a `narrative` link. Strength is 0.2 per shared retrieval, capped at 0.6 | Tang & Reagh 2026, in `lifecycle` |

## The tools

`tools/list` advertises these 16 tools, sorted by name. The list is compact: discriminator enums stay on the wire, and deep fields move one call deeper. The serialized catalog stays under 20 KiB. `memory_status` with `view=tools` lists every tool; set `tool` to a name for that tool's full input schema. Hidden aliases from older releases still dispatch, and they are not in the list.

| Tool | Purpose |
|---|---|
| `backfill` | Earlier candidates that share entities with a failure. Preview unless `promote` is true |
| `codebase` | Remember a pattern or decision, fetch context, verify anchors, reanchor evidence |
| `dedup` | Scan duplicates, plan a merge or supersede, apply, undo. Merged memories are invalidated |
| `graph` | Chains, associations, bridges, predictions, composition topology. `label` is the write |
| `intention` | Set, check, update, list. `graph` runs the evidence-aware plan |
| `maintain` | `consolidate`, `dream`, `gc` (dry run unless you turn it off), `importance_score`, `backup`, `export`, `restore` |
| `memory` | `get`, `get_batch`, `state`, `promote`, `demote`, `edit`, `purge` (`confirm=true`; retired, can't be retrieved). `delete` aliases `purge`. Demote does not delete |
| `memory_status` | `health`, `retention`, `timeline`, `changelog`, `stats`, `tools`. `view=tools` plus `tool` unfolds one full schema |
| `project` | Preview a fenced region of `CLAUDE.md` or `MEMORY.md`. `write` needs `confirm=true` and replaces only the fence |
| `purge` | Retire one memory so it can't be retrieved. `confirm=true` required. `destructiveHint` is true, and `_meta["anthropic/requiresUserInteraction"]` is true, so the client prompts. Same path as `memory` action `purge`. Returns the node id and an `eff-` receipt naming `purge` |
| `receipt` | `get` a stored receipt, or `replay` it with named slots withheld |
| `recall` | `lookup` (hybrid search), `reason`, or `contradictions`. Retrieval does not change strength |
| `session_start` | Memories, open intentions, status, predictions, and codebase context under one budget |
| `smart_ingest` | Create, merge, or supersede through prediction-error gating. `items` batches up to 20 |
| `source_sync` | Index GitHub (`GITHUB_TOKEN`) or Redmine (`REDMINE_URL`, `REDMINE_API_KEY`) into local memories. Opt-in (`--features connectors`); absent from a default `tools/list` |
| `suppress` | Hold a memory out of retrieval and speed its decay. `reverse=true` undoes it inside 24 hours when the stored snapshot still matches |

Full contracts: [docs/TOOL-CONTRACTS.md](docs/TOOL-CONTRACTS.md) · Hygiene and standing habits: [docs/MEMORY_HYGIENE.md](docs/MEMORY_HYGIENE.md)

## The dashboard

```bash
vestige dashboard
```

The server binds **http://127.0.0.1:3927** and redirects `/` to **/dashboard**. The observatory steps a fixed 60fps clock, 720 frames, 12 seconds, and can export that loop as an mp4. Share artifacts are structure-only: the shape of the store, not the memory text.

## Under the hood

| | |
|---|---|
| Engine | Rust 2024. Release archives ship `vestige`, `vestige-mcp`, `vestige-restore`, and `vestige-upgrade` |
| Retrieval | Nomic Embed Text v1.5, Matryoshka 768d truncated to 256d, USearch HNSW, SQLite FTS5. The background reranker is Jina Reranker v1 Turbo; until it loads, ranking stays BM25 |
| Storage | SQLite. SQLCipher is the optional `encryption` feature plus `VESTIGE_ENCRYPTION_KEY`, not the default build. See [docs/STORAGE.md](docs/STORAGE.md) |
| First run | About 130 MB for the embedding model and about 150 MB for the reranker. Keyword search and a `warming` block until the embedding runtime is ready; BM25 until the reranker loads. A release build hints when npm has a newer `vestige-mcp-server`, and does not update itself |

## Go deeper

[Getting Started](docs/GETTING-STARTED.md) · [FAQ](docs/FAQ.md) · [The Science](docs/SCIENCE.md) · [Configuration](docs/CONFIGURATION.md) · [Storage](docs/STORAGE.md) · [Tool contracts](docs/TOOL-CONTRACTS.md) · [Changelog](CHANGELOG.md)

## License

AGPL-3.0-only. Hosted Continuity (`vestige sync --cloud`) is a separate service.
