const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { execFileSync } = require('node:child_process');

test('the committed meta package carries no optionalDependencies', () => {
  const meta = require('../package.json');
  assert.ok(!meta.optionalDependencies, 'pins are injected at publish time, not committed (#220)');
});

test('no lifecycle scripts remain on the meta package', () => {
  const meta = require('../package.json');
  assert.ok(!meta.scripts || !meta.scripts.postinstall, 'postinstall must not run eagerly (#220)');
});

test('the inject script pins every platform package at the current version', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-inject-'));
  const pkg = { name: 'vestige-mcp-server', version: '9.9.9' };
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify(pkg));
  fs.copyFileSync(path.join(__dirname, '..', 'scripts', 'inject-platform-deps.js'), path.join(dir, 'inject.js'));
  const out = execFileSync(process.execPath, [path.join(dir, 'inject.js'), path.join(dir, 'package.json')], { encoding: 'utf8' });
  assert.match(out, /5 platform pins/);
  const injected = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
  assert.deepStrictEqual(
    Object.keys(injected.optionalDependencies).sort(),
    ['@vestige/mcp-darwin-arm64', '@vestige/mcp-darwin-x64', '@vestige/mcp-linux-arm64', '@vestige/mcp-linux-x64', '@vestige/mcp-win32-x64'],
  );
  for (const range of Object.values(injected.optionalDependencies)) {
    assert.strictEqual(range, '9.9.9');
  }
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
    const dir = path.join(__dirname, '..', '..', name.replace('@vestige/', ''));
    const manifest = require(path.join(dir, 'package.json'));
    assert.strictEqual(manifest.name, name);
    assert.deepStrictEqual(manifest.os, osList);
    assert.deepStrictEqual(manifest.cpu, cpuList);
    assert.ok(fs.existsSync(path.join(dir, 'bin')), `${name} must carry bin/`);
  }
});
