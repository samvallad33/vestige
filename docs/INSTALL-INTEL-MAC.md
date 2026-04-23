# Intel Mac Installation

The Intel Mac (`x86_64-apple-darwin`) binary links dynamically against a system
ONNX Runtime instead of a prebuilt ort-sys library. Microsoft is discontinuing
x86_64 macOS prebuilts after ONNX Runtime v1.23.0, so we use the
`ort-dynamic` feature to runtime-link against the version you install locally.
This keeps Vestige working on Intel Mac without waiting for a dead upstream.

## Prerequisite

Install ONNX Runtime via Homebrew:

```bash
brew install onnxruntime
```

## Install

```bash
# 1. Download the binary
curl -L https://github.com/samvallad33/vestige/releases/latest/download/vestige-mcp-x86_64-apple-darwin.tar.gz | tar -xz
sudo mv vestige-mcp vestige vestige-restore /usr/local/bin/

# 2. Point the binary at Homebrew's libonnxruntime
echo 'export ORT_DYLIB_PATH="'"$(brew --prefix onnxruntime)"'/lib/libonnxruntime.dylib"' >> ~/.zshrc
source ~/.zshrc

# 3. Verify
vestige-mcp --version

# 4. Connect to Claude Code
claude mcp add vestige vestige-mcp -s user
```

`ORT_DYLIB_PATH` is how the `ort` crate's `load-dynamic` feature finds the
shared library at runtime. Without it the binary starts but fails on the first
embedding call with a "could not find libonnxruntime" error.

## Building from source

```bash
brew install onnxruntime
git clone https://github.com/samvallad33/vestige && cd vestige
cargo build --release -p vestige-mcp \
  --no-default-features \
  --features ort-dynamic,vector-search
export ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
./target/release/vestige-mcp --version
```

## Troubleshooting

**`dyld: Library not loaded: libonnxruntime.dylib`** — `ORT_DYLIB_PATH` is not
set for the shell that spawned `vestige-mcp`. Claude Code / Codex inherits the
env vars from whatever launched it; export `ORT_DYLIB_PATH` in `~/.zshrc` or
`~/.bashrc` and restart the client.

**`error: ort-sys does not provide prebuilt binaries for the target
x86_64-apple-darwin`** — you hit this only if you ran `cargo build` without the
`--no-default-features --features ort-dynamic,vector-search` flags. The default
feature set still tries to download a non-existent prebuilt. Add the flags and
rebuild.

**Homebrew installed `onnxruntime` but `brew --prefix onnxruntime` prints
nothing** — upgrade brew (`brew update`) and retry. Older brew formulae used
`onnx-runtime` (hyphenated). If your brew still has the hyphenated formula,
substitute accordingly in the commands above.

## Long-term

Intel Mac will move to a fully pure-Rust backend (`ort-candle`) in Vestige
v2.1, removing the Homebrew prerequisite entirely. Track progress at
[issue #41](https://github.com/samvallad33/vestige/issues/41).
