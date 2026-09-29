# Intel Mac Installation

> **4.0: no ONNX Runtime needed.** Everything below about `ort-dynamic`,
> `ORT_DYLIB_PATH`, and Homebrew `onnxruntime` describes the 3.x embedder
> path. 4.0 carries no embedding runtime on any platform, so the Intel Mac
> (`x86_64-apple-darwin`) archive is self-contained: download
> `vestige-mcp-x86_64-apple-darwin.tar.gz` from
> [GitHub Releases](https://github.com/samvallad33/vestige/releases/latest),
> unpack, and connect `vestige-mcp` — the same flow as the [README](../README.md#install).
> There is no dylib to install, no env var to export, and no GUI-client `env`
> block to configure. The rest of this page is kept for 3.x history.

## 3.x history (retired)

The 3.x Intel Mac binary linked dynamically against a system ONNX Runtime
(`ort-dynamic`) because Microsoft discontinued x86_64 macOS prebuilts after
ONNX Runtime v1.23.0. It required `brew install onnxruntime`,
`ORT_DYLIB_PATH` pointing at `libonnxruntime.dylib` (including in the MCP JSON
`env` block for GUI clients, which do not inherit `.zshrc`), and source builds
with `--features ort-dynamic,vector-search`. All of that existed to run the
embedding model — which 4.0 removed along with vector search itself. If an old
3.x install fails with "could not find libonnxruntime", upgrade to 4.0 rather
than repairing the dylib path.

> If you are coming from a 3.x SQLite store, migrate it read-only first:
> `vestige migrate-to-strata --from <path>` — see
> [MIGRATING-TO-4.0.md](MIGRATING-TO-4.0.md).
