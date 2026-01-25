#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

const platform = os.platform();
const arch = os.arch();

const binaryName = platform === 'win32' ? 'vestige-mcp.exe' : 'vestige-mcp';
const binaryPath = path.join(__dirname, '..', 'bin', binaryName);

const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: 'inherit',
});

child.on('exit', (code) => {
  process.exit(code ?? 0);
});
