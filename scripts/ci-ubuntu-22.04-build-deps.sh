#!/usr/bin/env bash
# Packages needed to cargo-build vestige-mcp inside ubuntu:22.04.
# The GHA linux-gnu job runs in that container so the ELF links glibc 2.35.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  build-essential \
  pkg-config \
  cmake \
  python3 \
  perl \
  xz-utils \
  unzip \
  binutils \
  libssl-dev
rm -rf /var/lib/apt/lists/*
