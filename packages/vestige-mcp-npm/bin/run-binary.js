const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

function installBinary() {
  const installerPath = path.join(__dirname, '..', 'scripts', 'postinstall.js');
  const result = spawnSync(process.execPath, [installerPath], {
    encoding: 'utf8',
  });

  // An MCP server must reserve stdout for protocol messages. Keep the delayed
  // installation progress on stderr even when the first invoked command is
  // vestige-mcp.
  if (result.stdout) process.stderr.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);

  if (result.error) {
    throw new Error(`Failed to install Vestige binary: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(`Vestige binary installation exited with status ${result.status}`);
  }
}

function runBinary(name, displayName, args) {
  const binaryName = os.platform() === 'win32' ? `${name}.exe` : name;
  const binaryPath = path.join(__dirname, binaryName);

  if (!fs.existsSync(binaryPath)) {
    try {
      installBinary();
    } catch (err) {
      console.error(`Error: ${err.message}`);
      process.exit(1);
    }
  }

  if (!fs.existsSync(binaryPath)) {
    console.error(`Error: ${displayName} binary not found.`);
    console.error(`Expected at: ${binaryPath}`);
    console.error('');
    console.error('Try reinstalling: npm install -g vestige-mcp-server');
    process.exit(1);
  }

  const child = spawn(binaryPath, args, { stdio: 'inherit' });
  child.on('error', (err) => {
    console.error(`Failed to start ${displayName}:`, err.message);
    process.exit(1);
  });
  child.on('exit', (code) => process.exit(code ?? 0));
}

module.exports = { runBinary };
