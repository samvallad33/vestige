# Contributing to Vestige

Thank you for your interest in contributing to Vestige. This guide covers what you need to build, test and submit a change.

## Project Overview

Vestige is the Causal Proof Engine and the operating system for AI agents, written in Rust. Its kernel is Strata, an append-only, hash-chained, signed log. The engine (recall, ranking, pairing and explanation) uses zero vectors, zero string matching and zero RAG. A record is found by an exact handle and linked only by an edge the log recorded. Every output carries its proof: a memory id, an edge path or a receipt.

Do not add embeddings, similarity, fuzzy matching, BM25 or full-text search to the engine. A change that does will not be merged.

**Layout:**

```
vestige/
├── crates/
│   ├── strata/             # Append-only log: frames, segments, hash chain
│   ├── strata-kernel/      # Canonical FSRS fold, checkpoints, replay verification
│   ├── strata-gate/        # The gate every write passes
│   ├── strata-store/       # Gate-admitted store on the log
│   ├── strata-verify/      # Log and receipt verification (`vestige strata-verify`)
│   ├── strata-migrate/     # v3 vestige.db to Strata importer
│   ├── vestige-core/       # Shared types, storage trait, intention graph, projection, credential scanner
│   ├── vestige-mcp/        # MCP server, CLI, Axum dashboard, tool handlers
│   ├── vestige-upgrade/    # Runs the v3 import on first launch
│   └── vestige-spacetime/  # Standalone algorithm crate; no shipped binary depends on it
├── apps/
│   └── dashboard/          # SvelteKit + Three.js dashboard
├── packages/               # npm installer and per-platform binary packages
├── operator-lite/          # The Operator Lite gate (Python, stdlib only)
└── tests/                  # e2e, phase_1 and differential test crates
```

The `strata*` crates are excluded from the root Cargo workspace and have their own manifests. The root workspace members are `vestige-core`, `vestige-spacetime`, `vestige-mcp`, `vestige-upgrade` and the three test crates.

## Development Setup

### Prerequisites

- **Rust**: the toolchain is pinned in `rust-toolchain.toml`. `rustup` installs it on the first `cargo` call. See [rustup.rs](https://rustup.rs).
- **Node.js** (v22+) and **pnpm**: only for the dashboard. The release workflow uses Node 22 and pnpm 10.

### Getting Started

```bash
git clone https://github.com/samvallad33/vestige.git
cd vestige

# Build the dashboard (the server embeds it with include_dir!)
cd apps/dashboard && pnpm install && pnpm build && cd ../..

# Build the Rust workspace
cargo build
```

Nothing downloads at build time or at first run. The default build has no embedding model.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `VESTIGE_DATA_DIR` | Data directory for `vestige-mcp`, `vestige` and `vestige-restore`. `--data-dir` wins over it. Use a scratch directory when you test |
| `VESTIGE_HTTP_ENABLED`, `VESTIGE_DASHBOARD_ENABLED` | Turn the HTTP transport or the dashboard on or off. The stdio scripts under `scripts/` set both to `false` |

More variables are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Running Tests

CI runs these. Run the narrowest one that covers your change, then the full set before you open a PR.

```bash
# Default workspace
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace

# Workspace-excluded Strata crates, each with its own manifest
cargo test --manifest-path crates/strata/Cargo.toml
cargo test --manifest-path crates/strata-kernel/Cargo.toml
cargo test --manifest-path crates/strata-gate/Cargo.toml
cargo test --manifest-path crates/strata-store/Cargo.toml
cargo test --manifest-path crates/strata-verify/Cargo.toml
cargo test --manifest-path crates/strata-migrate/Cargo.toml

# Dashboard
cd apps/dashboard && pnpm check && pnpm test && pnpm build
```

The v3 SQLite engine is not linked into the shipped binaries. Its tests run behind feature flags, for example `cargo test -p vestige-mcp --features v3-engine --lib --tests` and `cargo test -p vestige-core --features bundled-sqlite,v3-engine,connectors,cloud-sync --lib`. CI lists the full set in `.github/workflows/ci.yml`.

## Building

```bash
# Debug build
cargo build -p vestige-mcp

# Release build with the embedded dashboard
cargo build --release -p vestige-mcp

# The release binary is at target/release/vestige-mcp
```

Release archives hold four binaries: `vestige-mcp`, `vestige`, `vestige-upgrade` and `vestige-restore`. Linux release assets are built on Ubuntu 22.04 (glibc 2.35), so `vestige-mcp` starts on Ubuntu 22.04 and Debian 12. musl is not a release target.

### Release Profile

The release profile uses `lto = true`, `codegen-units = 1`, `opt-level = "z"`, `panic = "abort"` and `strip = true` for minimum binary size.

## Publishing a GitHub Release

Only a tag of the form `vX.Y.Z` builds product binaries. Any other release on this repository (Operator Lite, benchmarks, launches) builds and uploads nothing. A `verify-latest` job runs `scripts/check-latest-release.sh` after every release event and fails the run if `releases/latest` could not serve the install URL in the README.

Version tags drive two independent workflows:

1. **Release** (`.github/workflows/release.yml`) builds and uploads the binaries. Do not fold registry publishing into its jobs.
2. **Publish MCP Registry** (`.github/workflows/publish-mcp-registry.yml`) authenticates with `mcp-publisher login github-oidc` and publishes the repo-root `server.json` to `https://registry.modelcontextprotocol.io`.

`server.json` version (and each `packages[].version`) must match the release tag with the leading `v` stripped. The release workflow also checks the versions in `package.json`, `apps/dashboard/package.json`, `packages/vestige-init/package.json`, `packages/vestige-mcp-npm/package.json`, `packages/vestige-mcpb/manifest.json` and `lhm.plugin.json`. An already-listed version is skipped rather than republished. Dry run: Actions, Publish MCP Registry, Run workflow, with `dry_run` enabled.

GitHub OIDC for `io.github.samvallad33/*` is automatic from this repository (`id-token: write` on the registry job). No PAT, no GitHub App and no extra org or repo trust binding is required. Nothing publishes to crates.io.

## Code Style

### Rust

```bash
# Format
cargo fmt --all

# Lint (zero warnings policy)
cargo clippy --workspace -- -D warnings
```

- Rust 2024 edition for the root workspace
- Standard `rustfmt` defaults
- All public items should have doc comments
- Tests go in `#[cfg(test)] mod tests` at the bottom of each file, or under a crate's `tests/` directory for stdio tests

### TypeScript/Svelte (Dashboard)

```bash
cd apps/dashboard
pnpm check    # Svelte type checking
pnpm test     # Vitest
```

## Project Structure

### vestige-core

Shared types and the storage trait. Its default build links no SQLite. Key modules:

| Module | Purpose |
|--------|---------|
| `storage/memory_store.rs` | The storage trait the server calls. The Strata backend lives in `vestige-mcp` (`strata_memory.rs`) |
| `intention_graph.rs` | The deterministic evidence-aware intention evaluator |
| `projection.rs` | Rendering of the fenced region that `project` writes |
| `security.rs` | The credential scanner behind the gate's secret refusal |
| `codebase/` | Code anchors and `verify_anchor`, which compares exact source spans |
| `composition.rs` | Outcome types for GhostLink weaves |
| `neuroscience/`, `advanced/`, `search/`, `fts.rs` | v3 engine code (spreading activation, prediction-error gating, hybrid search, FTS). Kept in the tree for the `legacy-sqlite` build. The default 4.x build does not use it for recall, ranking or pairing. `neuroscience/prospective_memory.rs` still backs intention triggers |

### vestige-mcp

The MCP server, the `vestige` CLI and the dashboard. Key modules:

| Module | Purpose |
|--------|---------|
| `server.rs` | MCP JSON-RPC server, the tool catalog, and the table of actions a Strata log withholds |
| `strata_memory.rs`, `strata_memory/ghostlink.rs` | The Strata storage backend and the GhostLink engine over recorded edges |
| `tools/` | One file per tool or action family |
| `attach.rs` | The single-writer lock and the owner-only socket other agents attach through |
| `dashboard/` | Axum HTTP, WebSocket and the request guards |
| `bin/cli.rs` | The `vestige` CLI |
| `main_sqlite.rs`, `bin/cli_sqlite.rs`, `cognitive.rs` | v3 engine entry points, behind `legacy-sqlite` |

### apps/dashboard

SvelteKit 2 + Three.js + Tailwind CSS. Pages are under `apps/dashboard/src/routes/(app)/`, for example `graph`, `memories`, `timeline`, `feed`, `explore`, `intentions`, `stats`, `dreams`, `blackbox` and `observatory`. Pages that need a feature the Strata log withholds show the reason code the server returns.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for new functionality
3. **Ensure all checks pass**: `cargo fmt`, `cargo clippy`, `cargo test`
4. **Build the dashboard** if you modified `apps/dashboard/`
5. **Keep commits focused**: one logical change per commit
6. **Open a PR** with a clear description

### PR Checklist

- [ ] `cargo fmt --all` formats the code
- [ ] `cargo clippy --workspace -- -D warnings` reports zero warnings
- [ ] `cargo test --workspace` passes, and so do the tests of any Strata crate you touched
- [ ] Dashboard builds (if modified): `cd apps/dashboard && pnpm build`
- [ ] No secrets, API keys or credentials in code
- [ ] A new claim in the docs is backed by a command you ran or code you read

### Good First Issues

Look for issues labeled `good first issue`. These are scoped, well-defined tasks:

- Adding tests for existing modules
- Documentation improvements
- Dashboard UI enhancements

## Adding or Changing a Tool

1. Create `crates/vestige-mcp/src/tools/your_tool.rs` with a schema and an `execute` function.
2. Register it in `crates/vestige-mcp/src/tools/mod.rs`, add its entry to `tool_catalog` in `server.rs`, and add its dispatch arm in `server.rs`.
3. Keep `tools/list` small. A build-time guard fails if the catalog exceeds 22 KiB. Put long descriptions in the full schema that `memory_status` with `view='tools'` serves.
4. If the action cannot work on a Strata log, add it to `STRATA_WITHHELD_ACTIONS` in `server.rs` with the reason. Do not return a zero result in its place. Reason codes are `unavailable_in_4_0`, `similarity_disabled` and `pending_strata`.
5. Every write goes through the gate and must return its receipt.
6. Add tests, including a stdio test under `crates/vestige-mcp/tests/` when the behavior is visible on the wire.
7. Update the tool count in the README and in `CLAUDE.md` and `AGENTS.md`.

## Issue Reporting

Use the issue templates:

- **Bug Report**: include OS, install method, IDE, Vestige version, and steps to reproduce
- **Feature Request**: describe the problem, the proposed solution, and alternatives considered

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to be respectful, constructive, and collaborative. Harassment and discrimination will not be tolerated.

## License

By contributing, you agree that your contributions will be licensed under **AGPL-3.0-only** ([LICENSE](LICENSE)), the same license as the project.

---

Questions? Open a [discussion](https://github.com/samvallad33/vestige/discussions) or reach out to the maintainers.
