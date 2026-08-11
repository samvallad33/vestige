const assert = require('assert/strict');
const { execFileSync, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const test = require('node:test');

const packageDir = path.resolve(__dirname, '..');

function run(command, args, options = {}) {
  return execFileSync(command, args, { encoding: 'utf8', ...options });
}

test('a pnpm global install exposes commands when lifecycle scripts are ignored', () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-pnpm-global-'));
  const archiveDir = path.join(tempDir, 'archive');
  const extractedDir = path.join(tempDir, 'extracted');
  const globalDir = path.join(tempDir, 'global');
  const globalBinDir = path.join(tempDir, 'bin');
  fs.mkdirSync(archiveDir);
  fs.mkdirSync(extractedDir);
  fs.mkdirSync(globalBinDir);

  try {
    const pack = JSON.parse(run('npm', ['pack', '--pack-destination', archiveDir, '--json'], { cwd: packageDir }));
    const packageArchive = path.join(archiveDir, pack[0].filename);
    run('tar', ['-xzf', packageArchive, '-C', extractedDir]);

    // Replace the release downloader in the packed fixture. This lets the
    // smoke test cover the pnpm topology without depending on GitHub assets.
    const fixtureInstaller = path.join(extractedDir, 'package', 'scripts', 'postinstall.js');
    fs.writeFileSync(
      fixtureInstaller,
      `const fs = require('fs');\n` +
        `const path = require('path');\n` +
        `const binDir = path.join(__dirname, '..', 'bin');\n` +
        `for (const name of ['vestige', 'vestige-mcp', 'vestige-restore']) {\n` +
        `  const binary = path.join(binDir, name);\n` +
        `  fs.writeFileSync(binary, '#!/bin/sh\\necho "fixture ' + name + ' $@"\\n');\n` +
        `  fs.chmodSync(binary, 0o755);\n` +
        `}\n` +
        `process.stdout.write('fixture installer ran\\n');\n`
    );
    const fixtureArchive = path.join(tempDir, 'vestige-mcp-server-fixture.tgz');
    run('tar', ['-czf', fixtureArchive, '-C', extractedDir, 'package']);

    const env = { ...process.env, PATH: `${globalBinDir}${path.delimiter}${process.env.PATH}` };
    const install = spawnSync(
      'pnpm',
      [
        'add',
        '--global',
        '--ignore-scripts',
        '--global-dir',
        globalDir,
        '--global-bin-dir',
        globalBinDir,
        fixtureArchive,
      ],
      { encoding: 'utf8', env }
    );
    assert.equal(install.status, 0, install.stderr || install.stdout);

    const cliPath = path.join(globalBinDir, 'vestige');
    const mcpPath = path.join(globalBinDir, 'vestige-mcp');
    assert.ok(fs.existsSync(cliPath), 'pnpm should link the vestige CLI');
    assert.ok(fs.existsSync(mcpPath), 'pnpm should link the vestige-mcp CLI');

    const cli = spawnSync(cliPath, ['--version'], { encoding: 'utf8', env });
    assert.equal(cli.status, 0, cli.stderr);
    assert.equal(cli.stdout, 'fixture vestige --version\n');
    assert.match(cli.stderr, /fixture installer ran/);

    const mcp = spawnSync(mcpPath, ['--version'], { encoding: 'utf8', env });
    assert.equal(mcp.status, 0, mcp.stderr);
    assert.equal(mcp.stdout, 'fixture vestige-mcp --version\n');
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
