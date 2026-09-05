# Intel Mac Installation

The Intel Mac (`x86_64-apple-darwin`) binary links dynamically against a system
ONNX Runtime instead of a prebuilt ort-sys library. Microsoft is discontinuing
x86_64 macOS prebuilts after ONNX Runtime v1.23.0, so we use the
`ort-dynamic` feature to runtime-link against the version you install locally.
This keeps Vestige working on Intel Mac without waiting for a dead upstream.

As of Vestige 2.7.1 this is still the Intel Mac path. Homebrew ONNX Runtime
remains required; there is no pure-Rust Intel backend in the current release.

## Prerequisite

Install ONNX Runtime via Homebrew:

```bash
brew install onnxruntime
```

## Install

```bash
# 1. Install the binary
npm install -g vestige-mcp-server@latest

# 2. Point the binary at Homebrew's libonnxruntime (CLI / terminal clients)
echo 'export ORT_DYLIB_PATH="'"$(brew --prefix onnxruntime)"'/lib/libonnxruntime.dylib"' >> ~/.zshrc
source ~/.zshrc

# 3. Verify
vestige-mcp --version

# 4. Connect to Claude Code (inherits the shell env if you launched it from a terminal)
claude mcp add vestige vestige-mcp -s user
```

`ORT_DYLIB_PATH` is how the `ort` crate's `load-dynamic` feature finds the
shared library at runtime. Without it the binary starts but fails on the first
embedding call with a "could not find libonnxruntime" error.

## GUI clients (Cursor, Claude Desktop)

Cursor and Claude Desktop **do not inherit `.zshrc`**. Exporting `ORT_DYLIB_PATH`
in your shell is not enough for those apps. Put it in the MCP JSON `env` block,
and paste the absolute `vestige-mcp` path from `which vestige-mcp`.

Run `brew --prefix onnxruntime` and paste the result. Do not hardcode
`/opt/homebrew` vs `/usr/local`.

```json
{
  "mcpServers": {
    "vestige": {
      "command": "<absolute path from which vestige-mcp>",
      "args": [],
      "env": {
        "ORT_DYLIB_PATH": "<brew --prefix onnxruntime>/lib/libonnxruntime.dylib"
      }
    }
  }
}
```

| Client | Config file |
|--------|-------------|
| Cursor | `~/.cursor/mcp.json` ([guide](integrations/cursor.md)) |
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` ([guide](CONFIGURATION.md#claude-desktop-macos)) |

## Building from source

```bash
brew install onnxruntime
git clone https://github.com/samvallad33/vestige && cd vestige
cargo build --release -p vestige-mcp \
  --no-default-features \
  --features ort-dynamic,vector-search,cloud-sync,connectors
export ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
./target/release/vestige-mcp --version
```

## Troubleshooting

**`dyld: Library not loaded: libonnxruntime.dylib`** — `ORT_DYLIB_PATH` is not
set for the process that spawned `vestige-mcp`. Terminal clients pick it up from
`~/.zshrc` / `~/.bashrc`. Cursor and Claude Desktop do not: put `ORT_DYLIB_PATH`
in the MCP JSON `env` block (see above) and restart the app.

**`error: ort-sys does not provide prebuilt binaries for the target
x86_64-apple-darwin`** — you hit this only if you ran `cargo build` without the
`--no-default-features --features ort-dynamic,vector-search` flags. The default
feature set still tries to download a non-existent prebuilt. Add the flags and
rebuild.

**Homebrew installed `onnxruntime` but `brew --prefix onnxruntime` prints
nothing** — upgrade brew (`brew update`) and retry. Older brew formulae used
`onnx-runtime` (hyphenated). If your brew still has the hyphenated formula,
substitute accordingly in the commands above.
