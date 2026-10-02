<p align="center">
  <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-logo.png" alt="Vestige" width="620">
</p>

# Vestige

**Local memory for MCP agents, on a signed log.**

It keeps the decisions a project already made, and it can walk a failure backward along the links your memory actually recorded. Every write passes a gate and leaves a receipt. The store is Strata: an append-only, signed log on your machine.

<p align="center">
  <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">
    <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/black-box-cause.gif" alt="Vestige Black Box — the cause never looks like the bug" width="100%">
  </a>
</p>

**New in 4.1.0:** [GhostLink](#ghostlink-the-negative-space) proposes pairs of memories nobody has combined yet, each with its proof, and 19 hardening fixes cover data safety, upgrades from v3, the tools and the dashboard ([changelog](CHANGELOG.md)).

[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/github/actions/workflow/status/samvallad33/vestige/ci.yml?branch=main&label=CI)](https://github.com/samvallad33/vestige/actions)
[![Binary](https://img.shields.io/badge/platforms-5_release_targets-informational)](https://github.com/samvallad33/vestige/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6)](LICENSE)

[Install](#install) · [Upgrading from v3](#upgrading-from-v3) · [Recall by handle](#recall-by-handle-not-resemblance) · [GhostLink](#ghostlink-the-negative-space) · [The tools](#the-tools) · [Backups](#backups-and-export) · [Science](#the-science) · [Docs](#go-deeper)

<a id="getting-started"></a>
## 🛡️ Operator Lite — your agent's commands, enforced on the metal

Ships in this repo under [`operator-lite/`](operator-lite/): a one-file, stdlib-only,
dependency-free gate that sits between your AI agent and your machine as a
PreToolUse hook (Claude Code, Codex, OpenClaw — any host with command hooks), and
**blocks destructive, polluting and exfiltrating commands before they run.**

```bash
mkdir -p ~/.operator/gate
curl -sL https://raw.githubusercontent.com/samvallad33/vestige/main/operator-lite/operator-gate.py -o ~/.operator/gate/operator-gate.py
chmod 755 ~/.operator/gate/operator-gate.py
echo shadow > ~/.operator/mode        # shadow-first: log everything, block nothing, then flip
```

- **27 deterministic rules** — workspace armor, memory-store protection, destructive
  SQL, force-push, unreviewed publishes, paid deploys, reverse shells, cloud-metadata
  endpoints, shell-init poisoning, MCP argument exfil
- **Sees through shell obfuscation** — quote reassembly (`r''m`), `$IFS` expansion,
  `$(echo rm)` as program, ANSI-C `$'\x72m'`, base64-decoded pipelines, brace/glob
  expansion against the live filesystem, subshell time-bombs, session variables, cd
  tracking, heredocs, fork bombs (10 of 11 vendor gates failed these classes)
- **Every verdict gets a hash-chained receipt** — and `verify` walks the chain
- **Shadow-first**: install logs everything and blocks nothing until you flip

```bash
python3 ~/.operator/gate/operator-gate.py corpus guardfall   # 43/43 bypass cases
python3 ~/.operator/gate/operator-gate.py verify             # receipt chain check
```

**On OpenClaw?** One line, gate bundled:

```bash
clawhub install vestige-operator-lite
```

Vestige's memory kernel is also on ClawHub as a six-tool plugin —
[`clawhub install vestige-strata`](https://clawhub.ai/plugins/vestige-strata) —
signed append-only log, exact-handle recall, causal root-cause walks.

Full details in [operator-lite/README.md](operator-lite/README.md). The free tier
blocks. Operator — the same kernel extended with a full governance loop — is where
the product lives.

## The cause never looks like the bug

Agents re-learn the same lessons. They recommend a change you already tested and rejected, re-derive a fix that was already written down, and treat every session as if the last one never happened. Vestige is the local memory an MCP client calls while you work: `smart_ingest` stores, `recall` finds a memory by an exact handle, and `causal_walk` walks a failure backward along edges the log recorded. Memory strength follows FSRS scheduling, and every write comes back with a receipt you can replay against the log.

<p align="center">
  <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">
    <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/black-box-cause.gif" alt="Vestige Black Box, recorded on v3.1" width="100%">
  </a>
</p>

<p align="center"><sub>Recorded on v3.1 with <code>vestige backfill --contrast</code>. Since 4.0 the backward walk is <code>causal_walk</code>, which follows only recorded edges from an explicit start point. <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">Watch the walk</a>.</sub></p>

## Install

Each release archive holds four binaries: `vestige-mcp` (the MCP server your agents run), `vestige` (the CLI), `vestige-upgrade` (the v3 importer) and `vestige-restore`. Keep all four in one folder. No Docker, no signup, no compile step, and nothing downloads on first start.

**macOS and Linux.** Two commands put the binaries in `~/.local/bin`:

```bash
mkdir -p ~/.local/bin
curl -fsSL https://github.com/samvallad33/vestige/releases/latest/download/vestige-mcp-aarch64-apple-darwin.tar.gz | tar -xz -C ~/.local/bin
```

Use the archive for your machine:

| Machine | Archive |
|---|---|
| Mac with Apple silicon | `vestige-mcp-aarch64-apple-darwin.tar.gz` |
| Mac with Intel | `vestige-mcp-x86_64-apple-darwin.tar.gz` |
| Linux x86_64 (glibc 2.35+: Ubuntu 22.04, Debian 12 and newer) | `vestige-mcp-x86_64-unknown-linux-gnu.tar.gz` |
| Linux arm64 (same glibc floor) | `vestige-mcp-aarch64-unknown-linux-gnu.tar.gz` |
| Windows x86_64 | `vestige-mcp-x86_64-pc-windows-msvc.zip` |

Then check it:

```bash
vestige-mcp --version
```

It should print `vestige-mcp 4.1.0`. If the shell says command not found, `~/.local/bin` is not on your PATH yet: add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` (or `~/.bashrc`) and open a new terminal. If it prints an older version, an older install comes first on your PATH; `which -a vestige-mcp` lists them.

On a Mac, download with `curl` as above rather than a browser. A browser marks the files as quarantined and macOS then refuses to run them. If you already used a browser, clear the flag with `xattr -d com.apple.quarantine ~/.local/bin/vestige*`.

**Windows.** Download `vestige-mcp-x86_64-pc-windows-msvc.zip` from [the latest release](https://github.com/samvallad33/vestige/releases/latest), unzip all four `.exe` files into one folder, and add that folder to your PATH.

**Homebrew** (macOS and Linux): `brew install samvallad33/tap/vestige`.

Every archive has a `.sha256` file beside it on the release page. Do not install this version with npm: the npm package still serves 3.0.0.

**Connect your agents.** The MCP command is `vestige-mcp`:

| Client | Setup |
|---|---|
| Claude Code | `claude mcp add vestige vestige-mcp -s user` |
| Codex | `codex mcp add vestige -- vestige-mcp` |
| Cursor / VS Code / Windsurf | [docs/integrations/](docs/integrations/) |
| Claude Desktop, and any other app you start from the Dock or Start menu | the JSON below, with the full path from `which vestige-mcp` (desktop apps do not read your shell's PATH) |
| Cline / Continue / Zed / Goose | the JSON below, in that client's MCP settings |

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

**Use it from every agent at once.** Claude Code in three terminals, Cursor, Codex and Claude Desktop can all run Vestige at the same time on one machine. The first one to start serves the store and the others connect to it, so every agent reads and writes the same memory through one writer. When that first agent quits, one of the others takes over and keeps its session.

`vestige-mcp` also takes `--data-dir <PATH>`, `--http`, `--no-http`, and `--http-port <PORT>` (default 3928, and passing the flag turns HTTP on). HTTP binds to `127.0.0.1` unless `VESTIGE_HTTP_BIND` is set. `VESTIGE_AUTH_TOKEN` overrides the bearer token. `VESTIGE_HTTP_ALLOWED_ORIGINS` is a comma-separated browser allowlist. `VESTIGE_DASHBOARD_ENABLED=1` starts the dashboard from the server; `VESTIGE_DASHBOARD_PORT` defaults to 3927. `VESTIGE_SYSTEM_PROMPT_MODE` is `minimal` or `full`. `RUST_LOG` filters logs. `VESTIGE_DATA_DIR` is the data directory when `--data-dir` is absent. The Strata log lives in `log/` inside that directory.

Optional output defaults live in `<data-dir>/vestige.toml`. A missing file uses the built-in defaults. An explicit MCP argument wins over the file.

```toml
[defaults]
profile = "default"       # lean | default | audit | research
detail_level = "summary"  # brief | summary | full
limit = 10
```

`lean` presets brief detail and a limit of 5. `audit` presets full detail. `research` presets full detail and a limit of 25. `default` leaves the historical tool limits alone.

Nothing leaves the machine by default. A default 4.x build has no embedding model and no startup version check. `source_sync` (`--features connectors`) and `vestige sync --cloud` (`--features cloud-sync`) are the only calls that use the network, and neither feature is in a 4.x release build. Full walkthrough: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

`vestige --help` lists the CLI. `--data-dir` is global.

<a id="upgrading-from-v3"></a>
## Upgrading from v3

Quit every app that runs Vestige v3 (each agent, the dashboard, any background job), point all of them at the 4.x `vestige-mcp`, and start them again. A v3 server left running keeps writing to `vestige.db`, which 4.x no longer reads after the upgrade.

Point 4.x at your existing data directory. The first launch finds `vestige.db`, builds a Strata log from it next to the file, verifies that log against a signed migration receipt, and only then publishes it as `log/`. The v3 file is never modified. A backup copy is written first, owner-only.

What carries over: every memory with its scope, tags and scheduling state; links; supersession; suppression (suppressed memories stay hidden); intentions; and code anchors. Links v3 inferred by similarity come across as `legacy_inferred` history. They are kept, and they never count as recorded evidence.

On a real 297 MB store with 8,902 memories in 34 scopes, the first launch took about 17 seconds before the MCP handshake answered, and average retention right after the upgrade was 0.810 against 0.8105 computed by v3 itself. Agents that start during the upgrade wait for it and then connect; there is nothing to coordinate by hand.

If the upgrade fails, the v3 data is untouched and the message says so. A `vestige.db` that is unreadable, empty or not plain SQLite is refused by name, never treated as an empty store, and a symlinked `vestige.db` is followed to the real file so memories that live only in its WAL come across. You can keep using v3.1.1 meanwhile. `vestige strata-verify <data-dir>` checks the log and its migration receipt at any time.

If you installed v3 with npm, make sure your agents now run the 4.x binary: `vestige-mcp --version` should print 4.1.0, and `which -a vestige-mcp` shows every copy on your PATH in the order they are found.

<a id="recall-by-handle-not-resemblance"></a>
## Recall by handle, not resemblance

Since 4.0, Vestige does not rank text that resembles your query. There are no embeddings, no BM25 and no keyword search in the default binaries. `recall` takes a handle: a memory id, a unique id prefix of 8 or more characters, or an exact tag. A free-text query returns `similarity_disabled` and asks for a handle.

| | Resemblance search | Vestige 4.x |
|---|---|---|
| How a memory is found | Similarity to the query | An exact handle |
| What counts as a link | Anything that scores close | Only an edge the log recorded. Imported v3 links are marked `legacy_inferred` |
| Walking back from a failure | Nearest lookalikes | `causal_walk` from an explicit logged write (a memory id), backward over recorded edges, at most 8 hops and 500 nodes by default. Failing-test, stack-frame, CI-run and version-range starts walk on a v3 store |
| Proof of a write | None | A receipt per write. `receipt replay` re-derives the state from the log |
| Unused memories | Persist at full weight | Fade under FSRS scheduling |
| Your data | Often a hosted index | A signed, append-only log in the data directory |

`causal_walk` never guesses. With no start point it returns `needs_report` and names what is missing. `forgotten_lesson` walks backward from a failure the same way and ranks fix or lesson memories by how far they have faded.

<a id="ghostlink-the-negative-space"></a>
## GhostLink: the negative space

`ghostlink` finds pairs of memories nobody has combined yet and hands one to your agent to compose. Every pair comes with its proof, built only from what the log recorded: ids, exact tags and types, typed edges, woven outcomes and FSRS state. No text, embeddings or keyword overlap.

- **Bridge lens** (the default): two memories within three hops over recorded `touched`, `derived_from` or `closed_by` edges that were never woven together. The shortest path is the proof. Pairs rank by hop proximity, how rarely each memory has been composed, retention, and the outcomes earlier pairs recorded.
- **Divergent lens**: two memories joined by no recorded edge at all, scored `min(path length, 7) × divergence`, where divergence is how few typed neighbors they share. When a memory has no typed neighbors there is nothing to measure, so the pair is a *forced juxtaposition* picked by a deterministic sampler, with no invented score. Links imported from v3 can only lower a score, never raise one.
- **Weave** records what came of a pair (`helpful`, `dead_end`, `accepted`, …): a composition memory plus a `derived_from` edge to each member, each with a receipt. Later proposals learn from it. A weave can carry `evidence` from outside: the URL, the sha256 of what was fetched, and when. Each finding is tagged `evidence:<sha256>`, and Vestige never fetches the URL itself.

```json
{"mode": "propose", "lens": "divergent", "limit": 5}
```

`map`, `inspect`, `explore`, `bounty`, `predict` and `harden` read and seed around the same graph. Reading never writes to the log.

<a id="founding-operator"></a>
## 🛡️ Every write is admitted

On a Strata log a write is proposed, checked by the gate, and admitted as an effect, and the call returns an `eff-` receipt. `receipt get` shows what that write did. `receipt replay` rebuilds the state from the log and reports any mismatch. `selftest` plants a known cause in a throwaway copy of your store and checks that the walk finds it, without touching the live store.

The Memory PR review modes from v3 (`risk_gated`, `paranoid`) are not available on a Strata log in 4.x. A review setting carried over from v3 reads as `fast`, and the dashboard's Memory PR list says review is unavailable.

**Recorded run on v3.1: [THE LIVE GATE](https://github.com/samvallad33/vestige/releases/tag/launch-night-live-gate-20260914).**

<a id="managed-continuity"></a>
<a id="backups-and-export"></a>
## 🔄 Backups and export

`maintain` action `backup` copies the whole Strata log into `<data-dir>/backups/` and reports its size. From a terminal, `vestige backup <new-folder>` does the same, and it works while your agents are running: it asks their Vestige server for the copy. To restore, stop Vestige and copy the backup's `log/` back into the data directory. Backups are owner-only (folders `0700`, files `0600`), and a log that fails verification is refused rather than copied. `session_start` reports `needsBackup` and `needsDream` from the last backup and dream that actually completed.

The log has one writer. CLI commands that open it directly, such as `vestige stats` or `vestige ingest`, run only while no Vestige server holds the store, and they name the process that does. `maintain` action `export` writes every live memory as JSON or JSONL into `<data-dir>/exports/`.

Portable archives, file sync and hosted sync are not available on a Strata log in 4.x. The CLI says so when you call them.

<a id="vestige-pro"></a>

## The receipts: Silent Rotation

Measured on v3.1, where recall still used hybrid search:

| Arm (6 models, 25 trials) | Converged correct | Converged wrong | Split |
|---|---|---|---|
| No memory | 0/25 | **21/25** | 4/25 |
| Dense cosine RAG | 4/23 | **12/23** | 7/23 |
| Vestige | 20/23 | **0/23** | 3/23 |

Outcomes come from `tests/by_model_tables.py`: a trial is correct when the tests were green, the production replay passed, and the key was right; wrong when the tests were green but production failed; split when the merge conflicted.

## The science

Write-up: [docs/SCIENCE.md](docs/SCIENCE.md).

| Mechanism | In 4.x | Source in the code |
|---|---|---|
| FSRS scheduling | Every card, native or imported, decays and strengthens under FSRS in the Strata kernel | `strata-kernel` `fsrs` |
| Memory dreaming | `maintain` actions `dream` and `dream_compile` replay recorded edges and strengthen co-activated ones | `dream`, `dream_compile` |
| Active forgetting | `suppress` takes a memory out of every read and keeps its bytes | `suppress` |
| Causal walk | Backward over recorded edges only, bounded and deterministic | `causal_walk` |
| Prediction-error gating, synaptic tagging, spreading activation, narrative edges, Retroactive Salience Backfill | In the v3 engine. The 4.x default build does not use them, because each depends on similarity or inferred links | `vestige-core` (v3 engine) |

## The tools

`tools/list` advertises these 16 tools, sorted by name. The list is compact: every field a call can send is on the wire with its type, prose and deep structure move one call deeper, and filter fields are grouped under `filters` and `source` (sent either grouped or flat). The serialized catalog is about 20 KiB. `memory_status` with `view=tools` lists every tool; set `tool` to a name for that tool's full input schema. Actions a Strata log cannot honor are left out of the schema and refused with the reason.

| Tool | Purpose |
|---|---|
| `causal_walk` | Walk a failure backward from explicit start points over recorded edges |
| `codebase` | Remember a pattern or decision with code anchors, fetch context marked current or stale, `verify` anchors against source, `reanchor` |
| `dedup` | `scan` for duplicates, `undo` a recorded operation, `tag_rename` and `tag_merge` with a preview, `policy` |
| `forgotten_lesson` | Faded fix or lesson memories behind a failure, over recorded edges |
| `ghostlink` | Never-composed memory pairs, each with its proof from recorded edges only. `propose` with lens `bridge` (within three typed-edge hops, never woven) or `divergent` (no recorded edge at all; a forced juxtaposition when nothing can be measured), `weave` an outcome (a write with receipts), `inspect` woven compositions, `explore` typed paths, `map`, `bounty`, `predict`, and `harden` to seed invariant laws once. `graph` still answers as a hidden alias |
| `intention` | `set`, `check`, `update`, `list`. `graph` runs the evidence-aware plan |
| `maintain` | `consolidate`, `dream`, `dream_compile`, `gc` (dry run unless you turn it off), `importance_score`, `backup`, `export` |
| `memory` | `get`, `get_batch`, `state`, `promote`, `demote`, `edit`. Demote does not delete. An edit admits a successor and keeps its code anchors |
| `memory_status` | `health`, `retention`, `timeline`, `changelog`, `provenance`, `coverage`, `stats`, `tools` |
| `project` | Preview a fenced region of `CLAUDE.md` or `MEMORY.md`. `write` needs `confirm=true` and replaces only the fence |
| `recall` | Find memories by exact handle: id, unique prefix, or exact tag |
| `receipt` | `get` a receipt, or `replay` it against the log |
| `selftest` | Plant a known cause in a throwaway copy and check the walk finds it |
| `session_start` | Status, open intentions for the scope (with id, due date and an overdue mark), backup and dream needs, and codebase context under one budget. It writes nothing. Queries are answered with a notice: memories are found by handle |
| `smart_ingest` | Store one memory, or up to 20 with `items`. `links` declares typed edges to existing memories (`derived_from`, `evidence_of`, `closes`), each with its own receipt, which is what GhostLink's bridge lens walks. Secrets are refused unless you say otherwise |
| `suppress` | Take a memory out of every read. The log keeps its bytes, and on Strata it cannot be undone. `destructiveHint` is true |

**Withheld since 4.0.** `purge` is the one tool 4.x does not ship. On an append-only signed log a purge could hide a memory but not erase its bytes, and a tool called purge must not pretend otherwise. `purge`, `memory` action `purge` or `delete`, and `delete_knowledge` return `unavailable_in_4_0`. Real erasure is planned as crypto-erasure ([#402](https://github.com/samvallad33/vestige/issues/402)).

Full contracts: [docs/TOOL-CONTRACTS.md](docs/TOOL-CONTRACTS.md) · Hygiene and standing habits: [docs/MEMORY_HYGIENE.md](docs/MEMORY_HYGIENE.md)

## The dashboard

```bash
vestige dashboard
```

The server binds **http://127.0.0.1:3927** and redirects `/` to **/dashboard**. It works while your agents are running: the Vestige server they share serves the dashboard until you press Ctrl+C. It answers only to `127.0.0.1` and `localhost` on its port. Only its own page can change memory; a script needs `Authorization: Bearer $VESTIGE_AUTH_TOKEN` ([Dashboard access](docs/CONFIGURATION.md#dashboard-access)). The observatory steps a fixed 60fps clock, 720 frames, 12 seconds, and can export that loop as an mp4. Share artifacts are structure-only: the shape of the store, not the memory text.

## Under the hood

| | |
|---|---|
| Engine | Rust 2024. Release archives ship `vestige`, `vestige-mcp`, `vestige-restore`, and `vestige-upgrade` |
| Store | Strata: an append-only log of borsh frames, hash-chained; a sealed segment carries a signed trailer. Writes are proposed, gated and admitted; the state is re-derived by replaying the log. No SQLite is linked into the shipped binaries |
| Recall | Exact handles only. No embeddings, BM25, FTS or keyword matching |
| Size | `vestige-mcp` is about 8.0 MB on macOS arm64 |
| First run | Nothing to download. On a v3 data directory, the first launch runs the upgrade before the MCP handshake |

## Go deeper

[Getting Started](docs/GETTING-STARTED.md) · [FAQ](docs/FAQ.md) · [The Science](docs/SCIENCE.md) · [Configuration](docs/CONFIGURATION.md) · [Storage](docs/STORAGE.md) · [Tool contracts](docs/TOOL-CONTRACTS.md) · [Changelog](CHANGELOG.md)

## License

AGPL-3.0-only.
