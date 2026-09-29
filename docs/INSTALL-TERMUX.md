# Android (Termux) Installation

Vestige runs on Android inside [Termux](https://termux.dev). Today that means
building from source; a prebuilt `aarch64-linux-android` release asset and
`npm install -g` support are the next step. Progress is on issue #145.

> **4.0 note.** There is no "semantic search through Termux's ONNX Runtime"
> step anymore: 4.0 carries no embedding runtime on any platform. `recall` is
> exact-handle only, dedup is exact content identity, and neither needs a
> model. The sections below describe what a Termux build actually ships.

## What you get in this build

- Every MCP tool, the memory lifecycle (FSRS scheduling, consolidation,
  suppression, purge), receipts, the dashboard, and `vestige`.
- Exact-handle recall and the full graph — no FTS, no model, nothing to warm up.

## What this build does not have

- Git history for the `codebase` tool. libgit2 is left out of this build, and the
  tool reports git history as unavailable instead of failing.

## Build

Termux packages a Rust toolchain built for `aarch64-linux-android`, so no cross
compiler is needed.

```bash
pkg install rust clang cmake make pkg-config git
git clone https://github.com/samvallad33/vestige.git
cd vestige
cargo build --release -p vestige-mcp --no-default-features --features connectors,cloud-sync
```

The binaries land in `target/release/`: `vestige-mcp`, `vestige` and
`vestige-restore`. Put them on your `PATH`:

```bash
install -m 755 target/release/vestige-mcp target/release/vestige target/release/vestige-restore "$PREFIX/bin/"
```

## Verify

```bash
vestige-mcp --version
vestige health
```

Then connect an MCP client with the usual config:

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

## Features in this configuration

| Feature | State | Why |
| --- | --- | --- |
| `connectors` | on | GitHub Issues and Redmine connectors; the HTTP client is rustls, no OpenSSL |
| `cloud-sync` | on | Vestige Pro sync client, same HTTP client |
| `embeddings`, `vector-search` | off (always-off stubs) | 4.0 removed the embedding machinery; the feature names exist only so legacy `#[cfg]` sites compile. There is nothing to turn on |
| `codebase-git` | off | libgit2 needs OpenSSL and libssh2 |

## Data location

Vestige stores its database under the XDG data directory, which on Termux
resolves through `$HOME` (`~/.local/share/vestige/`). Override with
`VESTIGE_DATA_DIR` or `--data-dir`.

## What is next

1. A prebuilt `aarch64-linux-android` asset in each release, and
   `npm install -g vestige-mcp-server` working on Termux.

(The old "semantic search on the phone" roadmap item is retired with the
embedding machinery itself.)

Follow or help on [issue #145](https://github.com/samvallad33/vestige/issues/145).
