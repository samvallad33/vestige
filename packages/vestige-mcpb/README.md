# Vestige MCPB

One-click installation bundle for Claude Desktop.

## For Users

1. Download `vestige-3.1.1.mcpb` from [GitHub Releases](https://github.com/samvallad33/vestige/releases)
2. Double-click to install
3. Restart Claude Desktop

That's it. No npm, no terminal, no config files.

## For Developers

### Building the bundle

```bash
# Install mcpb CLI
npm install -g @anthropic-ai/mcpb

# Download binaries from GitHub release
./build.sh

# Pack
mcpb pack
```

### Structure

```
vestige-mcpb/
├── manifest.json        # Bundle metadata
├── server/              # Platform binaries (downloaded)
│   ├── vestige-mcp-darwin-arm64
│   ├── vestige-upgrade-darwin-arm64
│   ├── vestige-mcp-linux-x64
│   ├── vestige-upgrade-linux-x64
│   ├── vestige-mcp-win32-x64.exe
│   └── vestige-upgrade-win32-x64.exe
└── vestige-3.1.1.mcpb  # Final bundle (generated)
```
