const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const POSIX = process.platform !== 'win32';
const INIT = path.join(__dirname, '..', 'bin', 'init.js');

function scratch(name) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `vestige-init-${name}-`));
}

function fakeBinary(dir, name, version) {
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, name);
  fs.writeFileSync(file, `#!/bin/sh\necho "${name} ${version}"\n`, { mode: 0o755 });
  return file;
}

function run(script, args, env) {
  const result = spawnSync(process.execPath, [script, ...args], {
    encoding: 'utf8',
    env: { PATH: '/usr/bin:/bin', ...env },
    timeout: 60000,
  });
  return { ...result, report: result.stdout.trim() ? JSON.parse(result.stdout) : null };
}

test('memory-first dry run previews installed binary connections and writes nothing', { skip: !POSIX }, () => {
  const home = scratch('dry-home');
  fs.mkdirSync(path.join(home, '.claude'));
  fs.mkdirSync(path.join(home, '.cursor'));
  fs.writeFileSync(path.join(home, '.cursor', 'mcp.json'), '{}\n');
  const bundled = scratch('dry-bundle');
  for (const name of ['vestige-mcp', 'vestige', 'vestige-restore', 'vestige-upgrade']) fakeBinary(bundled, name, '6.2.0');
  const target = path.join(home, '.local', 'bin');

  const { status, report, stderr } = run(INIT, ['--dry-run', '--json', '--client', 'cursor'], {
    HOME: home,
    VESTIGE_BUNDLED_BIN_DIR: bundled,
    VESTIGE_INSTALL_DIR: target,
    VESTIGE_MCP_PATH: fakeBinary(bundled, 'vestige-mcp', '6.2.0'),
  });
  assert.equal(status, 0, stderr);
  assert.equal(report.status, 'preview');
  assert.deepEqual(report.detected, ['cursor']);
  assert(report.files.some(row => row.file === path.join(home, '.cursor', 'hooks.json')));
  assert.equal(fs.existsSync(target), false, 'no binary is copied in a dry run');
  assert.equal(fs.readFileSync(path.join(home, '.cursor', 'mcp.json'), 'utf8'), '{}\n', 'no client config is written in a dry run');
  assert.equal(fs.existsSync(path.join(home, '.claude', 'settings.json')), false, 'no hook is written in a dry run');
});

test('init.js on its own, as scripts/setup.sh downloads it, still connects clients', { skip: !POSIX }, () => {
  const dir = scratch('standalone');
  const script = path.join(dir, 'init.js');
  fs.copyFileSync(INIT, script);
  const home = path.join(dir, 'home');
  fs.mkdirSync(path.join(home, '.cursor'), { recursive: true });
  fs.writeFileSync(path.join(home, '.cursor', 'mcp.json'), '{}\n');
  const binary = fakeBinary(path.join(dir, 'bin'), 'vestige-mcp', '5.0.0');

  const { status, report, stderr } = run(script, ['--client', 'cursor', '--json', '--no-dashboard'], {
    HOME: home,
    VESTIGE_MCP_PATH: binary,
    VESTIGE_INIT_VERSION: '5.0.0',
  });
  assert.equal(status, 0, stderr);
  assert.equal(report.binaries.status, 'kept');
  assert.equal(report.configured, 1);
  assert.equal(report.store, undefined, 'the clients-only wizard does not start the store');
  assert.equal(report.rules.status, 'skipped');
  assert.equal(report.gate.status, 'skipped');
  const written = JSON.parse(fs.readFileSync(path.join(home, '.cursor', 'mcp.json'), 'utf8'));
  assert.equal(written.mcpServers.vestige.command, binary);
});
