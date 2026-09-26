const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

// #220: the platform binary normally arrives as an exact-pinned
// optionalDependency (@vestige/mcp-<os>-<arch>), so installs need no
// lifecycle script at all — pnpm v10 and current Yarn block postinstall by
// default, and a registry-fetched binary also works behind corporate
// proxies that cannot reach GitHub Releases. The GitHub download remains
// only as a lazy fallback for `npm install --no-optional` and unusual
// platforms (esbuild#1647 documents the --no-optional gap).
function platformPackageName() {
  const platform = os.platform();
  const arch = os.arch();
  if (platform === 'darwin') return arch === 'arm64' ? '@vestige/mcp-darwin-arm64' : '@vestige/mcp-darwin-x64';
  if (platform === 'linux') return arch === 'arm64' ? '@vestige/mcp-linux-arm64' : '@vestige/mcp-linux-x64';
  if (platform === 'win32' && arch === 'x64') return '@vestige/mcp-win32-x64';
  return null;
}

function resolveFromPlatformPackage(name) {
  const pkg = platformPackageName();
  if (!pkg) return null;
  for (const suffix of ['bin', '.']) {
    try {
      const base = path.dirname(require.resolve(`${pkg}/package.json`));
      const candidate = path.join(base, 'bin', name);
      if (fs.existsSync(candidate)) return candidate;
    } catch (_) {
      // Platform package not installed (npm skips non-matching os/cpu).
    }
  }
  return null;
}

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

  // Resolution order: local dev checkout, platform package, lazy download.
  const candidates = [
    path.join(__dirname, binaryName),
    resolveFromPlatformPackage(binaryName),
  ].filter(Boolean);

  let binaryPath = candidates.find((candidate) => fs.existsSync(candidate));

  if (!binaryPath) {
    try {
      installBinary();
    } catch (err) {
      console.error(`Error: ${err.message}`);
      process.exit(1);
    }
    binaryPath = path.join(__dirname, binaryName);
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
