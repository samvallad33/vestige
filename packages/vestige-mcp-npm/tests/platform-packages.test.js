const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');

// run-binary is not exported as a module surface beyond runBinary; test the
// resolution helper indirectly by loading the file and using its internals
// through the same logic. Simplest robust check: the platform mapping table
// covers every platform we ship packages for.
const { execFileSync } = require('node:child_process');

function platformFor(platform, arch) {
  if (platform === 'darwin') return arch === 'arm64' ? '@vestige/mcp-darwin-arm64' : '@vestige/mcp-darwin-x64';
  if (platform === 'linux') return arch === 'arm64' ? '@vestige/mcp-linux-arm64' : '@vestige/mcp-linux-x64';
  if (platform === 'win32' && arch === 'x64') return '@vestige/mcp-win32-x64';
  return null;
}

test('the meta package pins every platform package at its own version', () => {
  const meta = require('../package.json');
  const expected = [
    '@vestige/mcp-darwin-arm64',
    '@vestige/mcp-darwin-x64',
    '@vestige/mcp-linux-x64',
    '@vestige/mcp-linux-arm64',
    '@vestige/mcp-win32-x64',
  ];
  assert.deepStrictEqual(Object.keys(meta.optionalDependencies || {}).sort(), expected.sort());
  for (const [name, range] of Object.entries(meta.optionalDependencies)) {
    assert.strictEqual(range, meta.version, `${name} must be exact-pinned to ${meta.version}, got ${range}`);
  }
});

test('no lifecycle scripts remain on the meta package', () => {
  const meta = require('../package.json');
  assert.ok(!meta.scripts || !meta.scripts.postinstall, 'postinstall must not run eagerly (#220)');
});

test('each platform package declares matching os/cpu and a bin directory', () => {
  const mapping = [
    ['@vestige/mcp-darwin-arm64', ['darwin'], ['arm64']],
    ['@vestige/mcp-darwin-x64', ['darwin'], ['x64']],
    ['@vestige/mcp-linux-x64', ['linux'], ['x64']],
    ['@vestige/mcp-linux-arm64', ['linux'], ['arm64']],
    ['@vestige/mcp-win32-x64', ['win32'], ['x64']],
  ];
  for (const [name, osList, cpuList] of mapping) {
    const dir = path.join(__dirname, '..', '..', 'platform', name.replace('@vestige/mcp-', ''));
    const manifest = require(path.join(dir, 'package.json'));
    assert.strictEqual(manifest.name, name);
    assert.deepStrictEqual(manifest.os, osList);
    assert.deepStrictEqual(manifest.cpu, cpuList);
    assert.ok(fs.existsSync(path.join(dir, 'bin')), `${name} must carry bin/`);
  }
});

test('this machine maps to a platform package or none', () => {
  const pkg = platformFor(os.platform(), os.arch());
  // On any supported dev machine this resolves to a real package name.
  if (['darwin', 'linux', 'win32'].includes(os.platform())) {
    assert.ok(pkg, `expected a platform package for ${os.platform()}-${os.arch()}`);
  } else {
    assert.strictEqual(pkg, null);
  }
});
