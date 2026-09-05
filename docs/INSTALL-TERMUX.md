# Android (Termux) Installation

Vestige runs on Android inside [Termux](https://termux.dev). Today that means
building from source in a configuration without embeddings; a prebuilt
`aarch64-linux-android` release asset and `npm install -g` support are the next
step, and semantic search through Termux's own ONNX Runtime package is the step
after that. This page tracks what works right now. Progress is on issue #145.

## What you get in this build

- Every MCP tool, the memory lifecycle (FSRS scheduling, consolidation,
  suppression, purge), receipts, the dashboard, and `vestige-cli`.
- Keyword recall (SQLite FTS5) and the full graph.

## What this build does not have

- Semantic recall. There is no embedding runtime, so `recall` is keyword only.
- The prediction-error gate. `smart_ingest` still stores every memory, but it
  cannot compare a new memory against existing ones by meaning, so automatic
  dedup and reinforce decisions are off. Responses say so
  (`"dedup": "unavailable in this build"`).
- Git history for the `codebase` tool. libgit2 is left out of this build, and the
  tool reports git history as unavailable instead of failing.

`vestige-cli health` reports `Embedding Service: not compiled into this build`
rather than "Not Ready", so the missing runtime is never mistaken for a broken
store.

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
| `embeddings`, `vector-search` | off | ONNX Runtime is not bundled for Android in this build |
| `codebase-git` | off | libgit2 needs OpenSSL and libssh2 |

## Data location

Vestige stores its database under the XDG data directory, which on Termux
resolves through `$HOME` (`~/.local/share/vestige/`). Override with
`VESTIGE_DATA_DIR` or `--data-dir`.

## What is next

1. A prebuilt `aarch64-linux-android` asset in each release, and
   `npm install -g vestige-mcp-server` working on Termux.
2. Semantic search on the phone: Termux ships ONNX Runtime as a package
   (`pkg install onnxruntime`), and Vestige already knows how to load a system
   `libonnxruntime` through `ORT_DYLIB_PATH` on Intel Macs. The Termux build
   will pick it up automatically.

Follow or help on [issue #145](https://github.com/samvallad33/vestige/issues/145).
