# Dockerfile for running the Vestige MCP server in an isolated sandbox.
#
# Used by registries such as Glama to start the server and run the standard
# MCP stdio introspection exchange (tools/list, resources/list, prompts/list).
# The server speaks MCP over stdio, which is exactly what these tools expect.
#
# Base must be glibc (Debian), not musl/Alpine: the npm postinstall downloads
# the prebuilt x86_64-unknown-linux-gnu Rust binary from the GitHub release, and
# a -gnu binary will not run on an Alpine/musl image.

FROM node:20-slim

# ca-certificates lets the postinstall fetch the release asset over HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the published package globally. Its postinstall downloads the matching
# prebuilt vestige-mcp binary for linux/x64 from the GitHub release.
RUN npm install -g vestige-mcp-server@latest

# Keep all memory data inside the container under a writable path.
ENV VESTIGE_DATA_DIR=/data
RUN mkdir -p /data

# Start the MCP server on stdio. The `vestige-mcp` bin execs the native binary
# and inherits stdio, so the MCP client talks to it directly.
ENTRYPOINT ["vestige-mcp"]
