#!/usr/bin/env bash
# Setup, then the Vestige side of the recording.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/setup.sh"
bash "$SCRIPT_DIR/vestige-side.sh"
