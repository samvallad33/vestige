# Vestige

**Local persistent memory and safety guardrails for AI agents — written in Rust, zero vectors, $0 API cost.**

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://github.com/samvallad33/vestige)
[![License: AGPL-3.0](https://img.shields.io/github/license/samvallad33/vestige)](LICENSE)
[![Release](https://img.shields.io/github/v/release/samvallad33/vestige)](https://github.com/samvallad33/vestige/releases/tag/v7.0.0) · [Live WebGPU Demo](https://stara-companion.pages.dev/?v=pro-reset)

## Install (1 Command)

```bash
npx @vestige/init
```

Detects Antigravity, Claude Code, Cursor, Codex, and Windsurf, configures the local MCP server, and wires lifecycle hooks. Restart your editor once done.

<details>
<summary>Standalone binaries, manual MCP JSON, Nix & CI Gate</summary>

- **macOS / Linux:** `curl -fsSL https://raw.githubusercontent.com/samvallad33/vestige/main/scripts/install.sh | sh`
- **Windows / binaries:** [GitHub Releases](https://github.com/samvallad33/vestige/releases/tag/v7.0.0)
- **Nix:** `nix run github:samvallad33/vestige`
- **Manual MCP JSON:** `{"mcpServers":{"vestige":{"command":"npx","args":["-y","vestige-mcp-server"]}}}`
- **GitHub PR Gate:** `uses: samvallad33/vestige@v7.0.0`

</details>

## Why Agents Use It

- **Zero-prompt recall:** Injects pinned rules, active plans, and past decisions automatically before every turn.
- **Live safety & anti-spiral:** Blocks risky shell commands, auto-repairs bad MCP parameters, and breaks failure loops by forcing root-cause inspection.
- **Verifiable local log:** Saves decisions and commits to an append-only, BLAKE3 hash-chained store (`Strata`) on your machine — no cloud DB, no embeddings.

## Tools

| Tool | Purpose |
| :--- | :--- |
| `session` | Boot session context, active plans, and live reflex state |
| `remember` | Save code-anchored decisions, facts, and fixes |
| `recall` | Deterministic handle and exact-token BM25 lookup |
| `causal_walk` | Trace root cause across failures, edits, and commits |
| `curate` | Edit, supersede, link, or hide stale memories |
| `precedent` | Inspect safety stops, scan secrets, verify receipts |

## Benchmarks (`v7.0.0`, CPU-Only)

| Benchmark | Metric | Score |
| :--- | :--- | :--- |
| **LongMemEval-S** (`470` q) | Session Any@5 / Any@10 | **97.87%** / **99.57%** |
| **LoCoMo-10** (`1,982` q) | Session R@5 / Turn Exact R@5 | **93.95%** / **83.55%** |
| **BEAM 100K** (`355` q) | Session Any@5 / Any@10 | **78.31%** / **87.32%** |
| **STATE-Bench** (`86` episodes) | Turn Exact R@5 / R@10 | **95.35%** / **100.0%** |

Reproduce locally: `bash benchmarks/run_all.sh` · Full tables: [benchmarks/README.md](benchmarks/README.md)

## Docs & License

[Storage & Backups](docs/STORAGE.md) · [Cognitive Sandwich](docs/COGNITIVE_SANDWICH.md) · [Changelog](CHANGELOG.md) · [AGPL-3.0](LICENSE) by [@samvallad33](https://github.com/samvallad33)
