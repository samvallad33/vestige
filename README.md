<p align="center">
  <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-logo.png" alt="Vestige" width="620">
</p>

# Vestige

**The Causal Proof Engine. The operating system for AI agents.**

**Zero vectors. Zero RAG. No lookalike text.** Nothing in Vestige's engine is found, ranked, paired or explained by embeddings, keyword scoring or similarity. Every answer comes from what the system recorded, and carries its proof: a memory id, an edge path, or a receipt. Two narrow exceptions are disclosed in [docs/TOOL-CONTRACTS.md](docs/TOOL-CONTRACTS.md): intention triggers match by substring, and a save can infer its start date from an "as of DATE" phrase.

Its kernel is **Strata**, an append-only, hash-chained, signed log on your machine. One install gives every agent on that machine the same kernel, the same gate, the same debugger and the same shell. Nothing leaves your machine by default.

Left alone, agents re-learn the same lessons. They recommend a change you already tested and rejected, re-derive a fix that was already written down, and run commands nobody checked. Vestige is the layer underneath them. It keeps what was decided, checks what is about to happen, proves what did, and when something breaks it walks the failure backward along the links the system recorded, because **the cause never looks like the bug.**

<p align="center">
  <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">
    <img src="https://raw.githubusercontent.com/samvallad33/vestige/media/black-box-cause.gif" alt="Vestige Black Box: the cause never looks like the bug" width="100%">
  </a>
</p>

<p align="center"><sub>Recorded on v3.1 with <code>vestige backfill --contrast</code>. Since 4.0 the backward walk is <code>causal_walk</code>, which follows only recorded edges. <a href="https://raw.githubusercontent.com/samvallad33/vestige/media/vestige-black-box.mp4">Watch the walk</a>.</sub></p>

[![Release](https://img.shields.io/github/v/release/samvallad33/vestige?color=06b6d4)](https://github.com/samvallad33/vestige/releases/latest)
[![Tests](https://img.shields.io/github/actions/workflow/status/samvallad33/vestige/ci.yml?branch=main&label=CI)](https://github.com/samvallad33/vestige/actions)
[![Binary](https://img.shields.io/badge/platforms-5_release_targets-informational)](https://github.com/samvallad33/vestige/releases/latest)
[![License](https://img.shields.io/badge/license-AGPL--3.0-3b82f6)](LICENSE)

[How it fits](#how-it-fits-together) · [What's new](#whats-new-in-411) · [Install](#install) · [The gate](#operator-lite) · [Recall by handle](#recall-by-handle-not-resemblance) · [GhostLink](#ghostlink-the-negative-space) · [The science](#the-science) · [System calls](#the-tools) · [Upgrading from v3](#upgrading-from-v3) · [Docs](#go-deeper)

<a id="how-it-fits-together"></a>
## How it fits together

Every layer is part of one system, and each does a job an agent cannot do for itself. The engine itself, meaning recall, ranking, pairing and explanation, uses no vectors, no RAG and no similarity; the Operator gate's command rules are a separate layer of pattern rules.

| Layer | What it is | What it does for your agents |
|---|---|---|
| **Kernel** | **Strata**: an append-only, hash-chained, signed log | The one source of truth. State is re-derived by replaying it. One process owns the store and every agent on the machine attaches to it |
| **Proof** | Receipts, typed edges, `receipt replay` | Every output names what justifies it. A link counts only if the log recorded it: a typed edge, a woven outcome, or a receipt |
| **Admission** | The gate on every write | A write is proposed, checked and admitted as an effect, and returns a receipt you can replay |
| **Addressing** | Exact handles: an id, a unique prefix, or an exact tag | An agent finds what it saved by name, not by lookalike wording |
| **Debugger** | `causal_walk`, `forgotten_lesson`, `selftest` | Walks a failure backward along recorded links to the decision behind it |
| **Lifecycle** | FSRS scheduling, `dream`, `suppress` | Records strengthen with use and fade without it. Dreaming replays recorded edges. Suppress takes a record out of every read |
| **Discovery** | GhostLink | Finds pairs of records nobody has combined, each with its proof |
| **Commitments** | `intention` and `session_start` | Deadlines and triggers surface when a session opens |
| **Source awareness** | `codebase` | Anchors decisions to code, flags drift, and turns git history into change records |
| **Security** | Operator Lite (free) and Operator (paid) | Blocks destructive commands before they run. Stdlib-only, with its own hash-chained receipts |
| **Shell** | `vestige dashboard` | See the store, its timeline and its receipts in a browser on `127.0.0.1` |
| **Snapshots** | `vestige backup`, `maintain` `export` | Owner-only copies, taken while your agents are running |

<a id="whats-new-in-411"></a>
## What's new in 4.1.1

4.1.1 makes the system more trustworthy at every layer. The debugger's walk now points the right way, a cause can be declared when it is saved, your git history becomes part of the system, and a call that cannot do something says so instead of returning zeros. Full list in the [changelog](CHANGELOG.md).

**Fixed: walks now go toward the cause.** A link written as `new derived_from existing` is stored in that direction, and GhostLink weaves write it too. `causal_walk` and `forgotten_lesson` followed it the wrong way, so from a symptom they returned the memories *derived from it* as its causes and never reached what the symptom itself derives from. They now go from a memory to what it derives from, to the records that are `evidence_of` it, that it closed, or that touched it. If you wove pairs on 4.1.0, walks across those edges now answer differently.

**New: declare the cause when you save.**
- `smart_ingest` takes typed `links` to existing memories: `derived_from`, `evidence_of` or `closes`. Each link is a recorded edge with its own receipt, checked before anything is written, and one bad link refuses the whole save. Those edges are what `causal_walk` follows and what GhostLink's bridge lens walks.
- `ghostlink` `weave` can carry up to 8 `evidence` findings from outside (URL, sha256 of what was fetched, time), tagged `evidence:<sha256>`. Vestige never fetches the URL, and the release binary stays offline.

**New: your git history becomes part of the system.** `codebase` action `ingest_repo` turns each non-merge commit of a local checkout into a change record in its own scope, anchored to every symbol and file it touched. `verify` then re-checks those records against your working tree, so you see which past changes the code has since drifted from. It previews first, skips anything the secret gate refuses, and re-running skips commits already recorded.

**Better walks.**
- `causal_walk` takes `node_id` on every start point (failing test, stack frame, CI run, version range). Several start points walk together, and `start_points` reports what happened to each one with a reason. The CLI gains `--node-id`.
- An empty walk says why: how many causal edges arrive from another scope, and how many incoming edges are not causal, by link type.
- `codebase` `get_context` lists every scope that holds a codebase's memories. `verify` says exactly what `limit` left unchecked. `session_start` names the scopes that hold code memories when the requested one has none.

**Honest responses.**
- A capability this build lacks says so. `maintain` `consolidate`, a no-op on Strata, is refused with `unavailable_in_4_0` instead of an all-zero "completed pass". The dashboard's Consolidate button answers the same. `dedup scan` and `dream` report `unavailable` with a reason.
- A successful `smart_ingest` no longer carries fields that read as a failure. A demote explains its own numbers: the memory is not deleted, and it fades faster.

**Security.**
- The credential gate now blocks Anthropic, OpenAI and Stripe live keys (`sk-ant-`, `sk-proj-`, `sk_live_` and the rest), in every stored field. `vestige scan-secrets` reports any already in the log. Stripe test keys are left alone.
- A hostile checkout's own git config can no longer make `ingest_repo` run a program.
- The credential scanner no longer aborts a save that ends within a few bytes of a key prefix such as `ghp_` or `AKIA`.

**Releases.** Binaries are built only for `vX.Y.Z` tags, and a check after every release fails the run if `releases/latest` could not serve the install URL below.

**Came with 4.1.0.** [GhostLink](#ghostlink-the-negative-space) proposes pairs of memories nobody has combined yet, each with its proof. The dashboard API now checks who is calling: before, any web page open in your browser could POST to it while it ran. A full disk is an error to the caller instead of a process abort, and a crashed owner can no longer block the store.

<a id="install"></a>
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

Check it:

```bash
vestige-mcp --version    # vestige-mcp 4.1.1
```

If the shell says command not found, add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` (or `~/.bashrc`) and open a new terminal. If it prints an older version, an older install comes first on your PATH; `which -a vestige-mcp` lists them. A running Vestige server keeps its old build until its app restarts, so restart every agent after an upgrade.

On a Mac, download with `curl` as above rather than a browser: a browser quarantines the files and macOS then refuses to run them. If you already used one, run `xattr -d com.apple.quarantine ~/.local/bin/vestige*`.

**Windows.** Download `vestige-mcp-x86_64-pc-windows-msvc.zip` from [the latest release](https://github.com/samvallad33/vestige/releases/latest), unzip all four `.exe` files into one folder, and add that folder to your PATH.

**Homebrew** (macOS and Linux): `brew install samvallad33/tap/vestige`.

Every archive has a `.sha256` file beside it on the release page. **npm** also works: `npm install -g vestige-mcp-server` (4.1.1 is published). Its installer downloads the same release archive for your platform and checks the `.sha256` before installing.

**Connect your agents.** The MCP command is `vestige-mcp`:

| Client | Setup |
|---|---|
| Claude Code | `claude mcp add vestige vestige-mcp -s user` |
| Codex | `codex mcp add vestige -- vestige-mcp` |
| Cursor / VS Code / Windsurf | [docs/integrations/](docs/integrations/) |
| Claude Desktop, and any app you start from the Dock or Start menu | the JSON below, with the full path from `which vestige-mcp` (desktop apps do not read your shell's PATH) |
| Cline / Continue / Zed / Goose | the JSON below, in that client's MCP settings |

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

**Every agent at once.** Claude Code in three terminals, Cursor, Codex and Claude Desktop can all run Vestige on one machine. The first to start serves the store and the others connect to it, so every agent reads and writes the same memory through one writer. When that first agent quits, another takes over and keeps its session.

**Configuration.** `vestige-mcp` takes `--data-dir`, `--http`, `--no-http` and `--http-port` (default 3928). HTTP binds to `127.0.0.1`. Environment variables, `vestige.toml` output profiles (`lean`, `default`, `audit`, `research`) and dashboard access are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md). `VESTIGE_SYSTEM_PROMPT_MODE` is `minimal` or `full`.

**Offline by default.** A default 4.x build has no embedding model and no startup version check. `source_sync` (`--features connectors`) and `vestige sync --cloud` (`--features cloud-sync`) are the only calls that use the network, and neither is in a 4.x release build. Walkthrough: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

<a id="operator-lite"></a>
<a id="getting-started"></a>
## The gate: Operator Lite

The security layer. Free, and in this repo under [`operator-lite/`](operator-lite/). A one-file, stdlib-only gate that sits between your agent and your machine as a PreToolUse hook, and blocks destructive, polluting and exfiltrating commands before they execute.

```bash
curl -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/operator-lite/operator-gate.py -o /tmp/operator-gate.py && python3 /tmp/operator-gate.py install
```

On Windows, in PowerShell, with Python 3.9 or newer:

```powershell
curl.exe -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/operator-lite/operator-gate.py -o "$env:TEMP\operator-gate.py"; python "$env:TEMP\operator-gate.py" install
```

One command copies the gate to `~/.operator/gate`, registers the Claude Code hook, and starts in **shadow mode**: it records every verdict and blocks nothing. It then replays your last 30 days of Claude Code history through the same rules, so the first thing you see is what it would have said about your own agents. Nothing in that history is executed. When the log looks right, `mode enforce` switches it on.

- **Deterministic rules, in two tiers.** 11 rules stop a command: protecting the gate itself, registered workspaces and the Vestige store, blind recursive delete, force-push over shared history, unreviewed publishes, paid deploys, destructive SQL, shell-init writes, reverse shells and cloud-metadata endpoints. 15 more run in shadow and flag without stopping: work-loss commands, outbound messages, public mutations, secret-shaped writes, pipe-to-interpreter, persistence writes, exfil shapes and MCP argument exfil among them.
- **Reads through shell obfuscation.** Quote reassembly, `$IFS` expansion, base64-decoded pipelines, brace and glob expansion, heredocs, subshell time-bombs.
- **Every verdict gets a hash-chained receipt**, and `verify` walks the chain.
- **Hosts.** Claude Code (`install` registers it for you), Codex and OpenClaw. [Ports](operator-lite/ports/) cover Cursor, Windsurf, Cline, Aider, Gemini CLI, Goose, OpenCode, OpenHands, Crush, Amazon Q, LangChain/LangGraph, CrewAI and the OpenAI Agents SDK.

```bash
python3 ~/.operator/gate/operator-gate.py status      # what it caught on your machine
python3 ~/.operator/gate/operator-gate.py replay      # re-run the 30-day history replay
python3 ~/.operator/gate/operator-gate.py verify      # receipt chain check
python3 operator-lite/operator-gate.py corpus guardfall   # 46/46 bypass cases, from a clone of this repo
```

On OpenClaw: `clawhub install vestige-operator-lite`. Vestige's kernel is also there as a six-tool plugin: [`clawhub install vestige-strata`](https://clawhub.ai/plugins/vestige-strata). Details: [operator-lite/README.md](operator-lite/README.md).

<a id="vestige-operator"></a>
**Vestige Operator** is the owner's version of the same gate: $149 once, every later version included. You write your own laws as sentences ("never publish without a dry run"), get a board of today's stops, a weekly digest of what your agents tried, and a five-minute onboarding wizard. [Buy Vestige Operator](https://payhip.com/b/d4xvu); from an installed Operator Lite, `upgrade --install <the archive you downloaded>` unpacks it and starts the wizard.

What it is not: Operator blocks what is routed through it. It cannot block a call that bypasses the hooked tools, and receipts are hash-chained digests, not signatures.

<a id="recall-by-handle-not-resemblance"></a>
## Recall by handle, not resemblance

Since 4.0, Vestige does not rank text that resembles your query. There are no embeddings, no BM25 and no keyword search in the default binaries. `recall` takes a handle: a memory id, a unique id prefix of 8 or more characters, or an exact tag. A free-text query returns `similarity_disabled` and asks for a handle.

Why: the cause of a bug rarely shares words with the error it produces. A search that ranks by lookalike words hands the agent the nearest-sounding memory, and the agent loads more context and guesses again. Vestige finds a memory by the handle you gave it and follows only the links the log recorded.

| | Resemblance search | Vestige 4.x |
|---|---|---|
| How a memory is found | Similarity to the query | An exact handle |
| What counts as a link | Anything that scores close | Only an edge the log recorded. Imported v3 links are marked `legacy_inferred` |
| Walking back from a failure | Nearest lookalikes | `causal_walk` from the memory that records the symptom, backward over recorded edges, at most 8 hops and 500 nodes by default |
| Proof of a write | None | A receipt per write. `receipt replay` re-derives the state from the log |
| Unused memories | Persist at full weight | Fade under FSRS scheduling |
| Your data | Often a hosted index | A signed, append-only log in the data directory |

`causal_walk` never guesses. With no start point it returns `needs_report` and names what is missing. A start point with no `node_id` is not walked, and the response says why. A walk that finds no cause says why in `emptyBecause`, from the edges the log holds. `forgotten_lesson` walks backward from a failure the same way and ranks fix or lesson memories by how far they have faded.

<a id="ghostlink-the-negative-space"></a>
## GhostLink: the negative space

`ghostlink` finds pairs of memories nobody has combined yet and hands one to your agent to compose. Every pair comes with its proof, built only from what the log recorded: ids, exact tags and types, typed edges, woven outcomes and FSRS state. No text, embeddings or keyword overlap.

- **Bridge lens** (default): two memories within three hops over recorded `touched`, `derived_from` or `closed_by` edges that were never woven together. The shortest path is the proof. Pairs rank by hop proximity, how rarely each memory has been composed, retention, and the outcomes earlier pairs recorded.
- **Divergent lens**: two memories joined by no recorded edge at all, scored by path length and how few typed neighbors they share. When a memory has no typed neighbors there is nothing to measure, so the pair is a *forced juxtaposition* picked by a deterministic sampler, with no invented score. Imported v3 links can only lower a score.
- **Weave** records what came of a pair (`helpful`, `dead_end`, `accepted`, ...) as a composition memory plus a `derived_from` edge to each member, each with a receipt. Later proposals learn from it, and a weave can carry outside `evidence`.

```json
{"mode": "propose", "lens": "divergent", "limit": 5}
```

`map`, `inspect`, `explore`, `bounty`, `predict` and `harden` read and seed around the same graph. Reading never writes to the log.

## Strata: every write is admitted and proven

Strata is the kernel. On a Strata log a write is proposed, checked by the gate, and admitted as an effect, and the call returns an `eff-` receipt. `receipt get` shows what that write did. `receipt replay` rebuilds the state from the log and reports any mismatch. `selftest` plants a known cause in a throwaway copy of your store and checks that the walk finds it, without touching the live store.

<a id="the-science"></a>
## The science

Vestige is built on memory research, and says plainly where it is inspired by it rather than a literal model of the brain. Write-up: [docs/SCIENCE.md](docs/SCIENCE.md).

| Mechanism | The idea from memory research | In 4.x | Source |
|---|---|---|---|
| FSRS scheduling | The forgetting curve: memories fade with time and strengthen when used | Every card, native or imported, decays and strengthens under FSRS in the Strata kernel | `strata-kernel` `fsrs` |
| Memory dreaming | Sleep replay: consolidation strengthens what was active together | `maintain` actions `dream` and `dream_compile` replay recorded edges and strengthen co-activated ones | `dream`, `dream_compile` |
| Active forgetting | Forgetting as a function, not a failure | `suppress` takes a memory out of every read and keeps its bytes | `suppress` |
| Causal walk | Tracing an effect back to its antecedents | Backward over recorded edges only, bounded and deterministic | `causal_walk` |
| Prediction-error gating, synaptic tagging, spreading activation, narrative edges, Retroactive Salience Backfill | Novelty gates storage; importance is assigned retroactively; activation spreads along associations | In the v3 engine. The 4.x default build does not use them, because each depends on similarity or inferred links | `vestige-core` (v3 engine) |

<a id="the-tools"></a>
## The tools: your agents' system calls

`tools/list` advertises 16 tools, sorted by name. The catalog is compact (about 20 KiB): every field a call can send is on the wire with its type, and prose moves one call deeper. `memory_status` with `view=tools` lists every tool; set `tool` to a name for its full input schema. Actions a Strata log cannot honor are left out of the schema and refused with the reason.

| Tool | Purpose |
|---|---|
| `causal_walk` | Walk a failure backward from explicit start points over recorded edges |
| `codebase` | Remember a pattern or decision with code anchors, fetch context marked current or stale, `verify` anchors against source, `reanchor`, and `ingest_repo` (commits as anchored change records) |
| `dedup` | `scan` for duplicates, `undo` a recorded operation, `tag_rename` and `tag_merge` with a preview, `policy` |
| `forgotten_lesson` | Faded fix or lesson memories behind a failure, over recorded edges |
| `ghostlink` | Never-composed memory pairs, each with its proof. `propose` (bridge or divergent lens), `weave`, `inspect`, `explore`, `map`, `bounty`, `predict`, `harden` |
| `intention` | `set`, `check`, `update`, `list`. `graph` runs the evidence-aware plan |
| `maintain` | `dream`, `dream_compile`, `gc` (dry run unless you turn it off), `importance_score`, `backup`, `export` |
| `memory` | `get`, `get_batch`, `state`, `promote`, `demote`, `edit`. Demote does not delete. An edit admits a successor and keeps its code anchors |
| `memory_status` | `health`, `retention`, `timeline`, `changelog`, `provenance`, `coverage`, `stats`, `tools` |
| `project` | Preview a fenced region of `CLAUDE.md` or `MEMORY.md`. `write` needs `confirm=true` and replaces only the fence |
| `recall` | Find memories by exact handle: id, unique prefix, or exact tag |
| `receipt` | `get` a receipt, or `replay` it against the log |
| `selftest` | Plant a known cause in a throwaway copy and check the walk finds it |
| `session_start` | Status, open intentions, backup and dream needs, and codebase context under one budget. It writes nothing |
| `smart_ingest` | Store one memory, or up to 20 with `items`. `links` declares typed edges to existing memories. Secrets are refused unless you say otherwise |
| `suppress` | Take a memory out of every read. The log keeps its bytes, and on Strata it cannot be undone |

**Withheld since 4.0.** `purge`, `memory` action `purge` or `delete`, and `delete_knowledge` return `unavailable_in_4_0`. On an append-only signed log a purge could hide a memory but not erase its bytes, and a tool called purge must not pretend otherwise. Real erasure is planned as crypto-erasure ([#402](https://github.com/samvallad33/vestige/issues/402)).

Full contracts: [docs/TOOL-CONTRACTS.md](docs/TOOL-CONTRACTS.md) · Habits: [docs/MEMORY_HYGIENE.md](docs/MEMORY_HYGIENE.md)

<a id="backups-and-export"></a>
## Backups and export

`vestige backup <new-folder>` copies the whole Strata log and works while your agents are running, because it asks their Vestige server for the copy. `maintain` action `backup` writes into `<data-dir>/backups/`. To restore, stop Vestige and copy the backup's `log/` back into the data directory. Backups are owner-only (folders `0700`, files `0600`), and a log that fails verification is refused rather than copied. `maintain` action `export` writes every live memory as JSON or JSONL into `<data-dir>/exports/`.

The log has one writer. CLI commands that open it directly, such as `vestige stats` or `vestige ingest`, run only while no Vestige server holds the store, and they name the process that does. Portable archives, file sync and hosted sync are not available on a Strata log in 4.x.

<a id="upgrading-from-v3"></a>
## Upgrading from v3

1. Quit every app that runs Vestige v3 (each agent, the dashboard, any background job).
2. Point them all at the 4.x `vestige-mcp` and start them again. A v3 server left running keeps writing to `vestige.db`, which 4.x no longer reads.
3. Use your existing data directory. The first launch builds a Strata log from `vestige.db`, verifies it against a signed migration receipt, and only then publishes it as `log/`. The v3 file is never modified, and an owner-only backup is written first.

Memories (with scope, tags and scheduling state), links, supersession, suppression, intentions and code anchors all carry over. Links v3 inferred by similarity come across as `legacy_inferred` history and never count as recorded evidence. On a real 297 MB store with 8,902 memories in 34 scopes, the first launch took about 17 seconds before the MCP handshake answered, and average retention was 0.810 afterward against 0.8105 computed by v3.

If the upgrade fails, the v3 data is untouched and you can keep using v3.1.1. `vestige strata-verify <data-dir>` checks the log and its migration receipt at any time. More in [docs/MIGRATING-v4.md](docs/MIGRATING-v4.md).

## The dashboard

```bash
vestige dashboard
```

Serves **http://127.0.0.1:3927** and works while your agents are running. It answers only to `127.0.0.1` and `localhost`, and only its own page can change memory; a script needs `Authorization: Bearer $VESTIGE_AUTH_TOKEN` ([Dashboard access](docs/CONFIGURATION.md#dashboard-access)). The observatory can export a 12-second loop of your memory graph as an mp4, and share artifacts are structure-only: the shape of the store, not the memory text.

## Under the hood

| | |
|---|---|
| Engine | Rust 2024. Release archives ship `vestige`, `vestige-mcp`, `vestige-restore`, and `vestige-upgrade` |
| Store | Strata: an append-only log of borsh frames, hash-chained; a sealed segment carries a signed trailer. Writes are proposed, gated and admitted; the state is re-derived by replaying the log. No SQLite is linked into the shipped binaries |
| Recall | Exact handles only. No embeddings, BM25, FTS or keyword matching |
| Size | `vestige-mcp` is about 8.0 MB on macOS arm64 |
| First run | Nothing to download. On a v3 data directory, the first launch runs the upgrade before the MCP handshake |

## Go deeper

[Getting Started](docs/GETTING-STARTED.md) · [FAQ](docs/FAQ.md) · [The Science](docs/SCIENCE.md) · [Configuration](docs/CONFIGURATION.md) · [Storage](docs/STORAGE.md) · [Tool contracts](docs/TOOL-CONTRACTS.md) · [Benchmarks](docs/BENCHMARKS.md) · [Changelog](CHANGELOG.md)

## License

AGPL-3.0-only.
