# Android (Termux) Installation

> Written for Vestige 4.2.0. Build from source: there is no Android release archive and no Android npm package. CI builds this target but nothing in CI runs it on a phone. Progress is on issue #145.

Vestige can run on Android inside [Termux](https://termux.dev). The 4.x engine has no embedding model, no ONNX Runtime and no vector index on any platform, so the only thing a Termux build leaves out is git history for the `codebase` tool.

## What you get in this build

The same engine as a default build, compiled without `codebase-git`: the Strata log, the gate and receipts, `recall` by exact handle, `causal_walk`, GhostLink, intentions, backups and the dashboard. Records are found by an id, a unique id prefix of 8 or more characters, or an exact tag.

## What this build does not have

- **Git history for the `codebase` tool.** libgit2 needs OpenSSL and libssh2, so this build leaves out the `codebase-git` feature. The tool reports git history as unavailable instead of failing.

## Build

Termux packages a Rust toolchain built for `aarch64-linux-android`, so no cross compiler is needed. The repository pins its toolchain in `rust-toolchain.toml`, and the comment there notes that the pinned version is the one Termux ships. CI builds the same target with `cargo-ndk` and `--no-default-features`, and checks that the binary needs only `libc`, `libm`, `libdl` and `liblog`.

```bash
pkg install rust clang cmake make pkg-config git
git clone https://github.com/samvallad33/vestige.git
cd vestige
cargo build --release -p vestige-mcp --no-default-features
cargo build --release -p vestige-upgrade
```

The binaries land in `target/release/`: `vestige-mcp`, `vestige`, `vestige-restore` and `vestige-upgrade`. Put them in one folder on your `PATH`:

```bash
install -m 755 target/release/vestige-mcp target/release/vestige target/release/vestige-restore target/release/vestige-upgrade "$PREFIX/bin/"
```

`vestige-upgrade` is the only binary that links SQLite. It imports an existing v3 `vestige.db` on first launch.

## Verify

```bash
vestige-mcp --version
```

It should print `vestige-mcp 4.1.1`. Then connect an MCP client with the usual config:

```json
{
  "mcpServers": {
    "vestige": { "command": "vestige-mcp" }
  }
}
```

To check the log itself, run `vestige strata-verify <data-dir>` while no Vestige server holds the store.

## Features in this configuration

| Feature | State | Why |
| --- | --- | --- |
| `connectors` | off | Not a 4.x default. Add `--features connectors` to opt in. See [CONNECTORS.md](CONNECTORS.md) |
| `cloud-sync` | off | Not a 4.x default. Add `--features cloud-sync` to opt in |
| `codebase-git` | off | libgit2 needs OpenSSL and libssh2 |
| `embeddings`, `vector-search` | no effect | They are empty stubs in 4.x and select no dependencies |

## Data location

Vestige stores its data in a per-user data directory. Override it with `VESTIGE_DATA_DIR` or `--data-dir`. The layout and the platform defaults are in [STORAGE.md](STORAGE.md).

Follow or help on [issue #145](https://github.com/samvallad33/vestige/issues/145).
