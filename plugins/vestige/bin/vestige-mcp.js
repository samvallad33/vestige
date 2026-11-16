#!/usr/bin/env node
'use strict';

// Claude Code starts this file as the plugin MCP server. A vestige-mcp binary
// already on PATH (GitHub release, Homebrew, or a local build) is used as-is.
// Otherwise npx installs vestige-mcp-server at the version in plugin.json.
// Stdout is the MCP transport, so every diagnostic goes to stderr.

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const manifestPath = path.join(__dirname, '..', '.claude-plugin', 'plugin.json');
const version = JSON.parse(fs.readFileSync(manifestPath, 'utf8')).version;

function onPath(name) {
  const dirs = (process.env.PATH || '').split(path.delimiter).filter(Boolean);
  const suffixes = process.platform === 'win32' ? ['.exe', '.cmd', '.bat', ''] : [''];
  for (const dir of dirs) {
    for (const suffix of suffixes) {
      const file = path.join(dir, name + suffix);
      try {
        if (fs.statSync(file).isFile()) return file;
      } catch {
        // keep searching
      }
    }
  }
  return null;
}

const binary = onPath('vestige-mcp');
const command = binary || (process.platform === 'win32' ? 'npx.cmd' : 'npx');
const args = binary
  ? process.argv.slice(2)
  : ['-y', `vestige-mcp-server@${version}`, ...process.argv.slice(2)];

if (!binary) {
  process.stderr.write(
    `vestige plugin: vestige-mcp is not on PATH; starting vestige-mcp-server@${version} with npx\n`
  );
}

const child = spawn(command, args, {
  stdio: 'inherit',
  windowsHide: true,
  // Node cannot spawn a .cmd without a shell. The argv is the plugin version
  // plus whatever Claude Code passed, not a shell string from the user.
  shell: !binary && process.platform === 'win32',
});
child.on('error', (error) => {
  process.stderr.write(`vestige plugin: failed to start MCP server: ${error.message}\n`);
  process.exit(1);
});
child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code == null ? 1 : code);
});
