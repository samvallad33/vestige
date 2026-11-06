#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
cargo build --release --locked -p vestige-mcp --bin vestige-mcp > runs/build.log 2>&1
.venv/bin/python -m benchmarks.real_engine.reporting --write-build-manifest
exec .venv/bin/python -m benchmarks.real_engine.run --autonomous "$@"
