const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const kernel = require('../lib/vestige-kernel.js');

const POSIX = process.platform !== 'win32';

function scratch(name) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `vestige-kernel-${name}-`));
}

// A stand-in binary that answers --version the way vestige-mcp does.
function fakeBinary(dir, name, version) {
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, name);
  fs.writeFileSync(file, `#!/bin/sh\necho "${name} ${version}"\n`, { mode: 0o755 });
  return file;
}

function bundle(version) {
  const dir = scratch('bundle');
  for (const name of kernel.BINARIES) fakeBinary(dir, name, version);
  return dir;
}

function withEnv(vars, fn) {
  const saved = {};
  for (const [key, value] of Object.entries(vars)) {
    saved[key] = process.env[key];
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return fn();
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const backup = fs.existsSync(file) ? `${file}.bak.test` : null;
  if (backup) fs.copyFileSync(file, backup);
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n');
  return backup;
}

test('each supported platform maps to the package that carries its binaries', () => {
  assert.equal(kernel.platformPackage('darwin', 'arm64'), '@vestige/mcp-darwin-arm64');
  assert.equal(kernel.platformPackage('darwin', 'x64'), '@vestige/mcp-darwin-x64');
  assert.equal(kernel.platformPackage('linux', 'arm64'), '@vestige/mcp-linux-arm64');
  assert.equal(kernel.platformPackage('linux', 'x64'), '@vestige/mcp-linux-x64');
  assert.equal(kernel.platformPackage('win32', 'x64'), '@vestige/mcp-win32-x64');
  assert.equal(kernel.platformPackage('win32', 'arm64'), null);
  assert.equal(kernel.platformPackage('freebsd', 'x64'), null);
});

test('versions are read from --version output and compared numerically', () => {
  assert.deepEqual(kernel.parseVersion('vestige-mcp 5.0.0\n'), [5, 0, 0]);
  assert.equal(kernel.parseVersion('no version here'), null);
  assert.ok(kernel.compareVersions([5, 0, 10], [5, 0, 9]) > 0);
  assert.ok(kernel.compareVersions([4, 9, 9], [5, 0, 0]) < 0);
  assert.equal(kernel.compareVersions([5, 0, 0], [5, 0, 0]), 0);
});

test('placing a binary over a symlink replaces the link and leaves its target alone', { skip: !POSIX }, () => {
  const dir = scratch('symlink');
  const pinned = fakeBinary(path.join(dir, 'pin'), 'vestige-mcp', '4.0.0');
  const link = path.join(dir, 'bin', 'vestige-mcp');
  fs.mkdirSync(path.dirname(link));
  fs.symlinkSync(pinned, link);
  const fresh = fakeBinary(path.join(dir, 'new'), 'vestige-mcp', '5.0.0');

  kernel.placeFile(fresh, link);

  assert.equal(fs.lstatSync(link).isSymbolicLink(), false);
  assert.match(fs.readFileSync(link, 'utf8'), /5\.0\.0/);
  assert.match(fs.readFileSync(pinned, 'utf8'), /4\.0\.0/, 'the pinned binary must not be overwritten');
});

test('a machine with no vestige gets all four binaries from the bundle', { skip: !POSIX }, () => {
  const target = scratch('install');
  const result = withEnv({ VESTIGE_BUNDLED_BIN_DIR: bundle('5.0.0'), VESTIGE_INSTALL_DIR: target },
    () => kernel.ensureBinaries({ found: null }));
  assert.equal(result.status, 'installed');
  assert.equal(result.version, '5.0.0');
  assert.equal(result.binaryPath, path.join(target, 'vestige-mcp'));
  assert.deepEqual(result.copied.sort(), [...kernel.BINARIES].sort());
  for (const name of kernel.BINARIES) assert.ok(fs.existsSync(path.join(target, name)), name);
});

test('an installed vestige-mcp as new as the bundle is kept, and nothing is copied', { skip: !POSIX }, () => {
  const target = scratch('keep');
  const found = fakeBinary(scratch('found'), 'vestige-mcp', '5.0.0');
  const result = withEnv({ VESTIGE_BUNDLED_BIN_DIR: bundle('5.0.0'), VESTIGE_INSTALL_DIR: target },
    () => kernel.ensureBinaries({ found }));
  assert.equal(result.status, 'kept');
  assert.equal(result.binaryPath, found);
  assert.deepEqual(fs.readdirSync(target), []);
});

test('an older vestige-mcp is replaced by the bundled one, and a dry run changes nothing', { skip: !POSIX }, () => {
  const target = scratch('upgrade');
  const found = fakeBinary(target, 'vestige-mcp', '4.2.0');
  const env = { VESTIGE_BUNDLED_BIN_DIR: bundle('5.0.0'), VESTIGE_INSTALL_DIR: target };

  const planned = withEnv(env, () => kernel.ensureBinaries({ found, dryRun: true }));
  assert.equal(planned.status, 'planned');
  assert.match(fs.readFileSync(found, 'utf8'), /4\.2\.0/);

  const result = withEnv(env, () => kernel.ensureBinaries({ found }));
  assert.equal(result.status, 'installed');
  assert.equal(result.replaced, '4.2.0');
  assert.match(fs.readFileSync(found, 'utf8'), /5\.0\.0/);
});

test('no bundle and no installed binary reports missing instead of guessing', () => {
  const result = withEnv({ VESTIGE_BUNDLED_BIN_DIR: path.join(scratch('empty'), 'nothing'), VESTIGE_INSTALL_DIR: scratch('none') },
    () => kernel.ensureBinaries({ found: null }));
  assert.equal(result.status, 'missing');
});

test('the rules hook is merged into Claude Code settings once, beside what is already there', () => {
  const home = scratch('home');
  const settingsPath = path.join(home, '.claude', 'settings.json');
  const vestige = fakeBinary(path.join(home, 'bin'), 'vestige', '5.0.0');
  const gate = { matcher: '*', hooks: [{ type: 'command', command: 'python3 ~/.operator/gate/operator-gate.py hook --source claude' }] };
  const other = { matcher: 'startup', hooks: [{ type: 'command', command: 'echo hello' }] };
  writeJson(settingsPath, { model: 'opus', hooks: { PreToolUse: [gate], SessionStart: [other] } });

  const first = kernel.wireRulesHook(vestige, { home, writeJson });
  assert.equal(first.status, 'written');
  const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
  assert.equal(settings.model, 'opus');
  assert.deepEqual(settings.hooks.PreToolUse, [gate]);
  assert.deepEqual(settings.hooks.SessionStart[0], other);
  const added = settings.hooks.SessionStart[1];
  assert.equal(added.matcher, kernel.RULES_MATCHER);
  assert.equal(added.hooks[0].command, `${vestige} hook session-start`);
  assert.equal(added.hooks[0].timeout, 10);
  assert.ok(first.backupPath && fs.existsSync(first.backupPath));

  const second = kernel.wireRulesHook(vestige, { home, writeJson });
  assert.equal(second.status, 'present');
  assert.equal(JSON.parse(fs.readFileSync(settingsPath, 'utf8')).hooks.SessionStart.length, 2);
});

test('a settings file that does not parse is left exactly as it is', () => {
  const home = scratch('broken');
  const settingsPath = path.join(home, '.claude', 'settings.json');
  fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
  const raw = '{ "hooks": { // mid-edit\n';
  fs.writeFileSync(settingsPath, raw);
  const vestige = fakeBinary(path.join(home, 'bin'), 'vestige', '5.0.0');

  const result = kernel.wireRulesHook(vestige, { home, writeJson });
  assert.equal(result.status, 'review');
  assert.equal(fs.readFileSync(settingsPath, 'utf8'), raw);
});

test('a dry run plans the rules hook and writes nothing; a path with spaces is quoted', () => {
  const home = scratch('dry');
  fs.mkdirSync(path.join(home, '.claude'), { recursive: true });
  const vestige = fakeBinary(path.join(home, 'Application Support', 'bin'), 'vestige', '5.0.0');

  const result = kernel.wireRulesHook(vestige, { home, dryRun: true, writeJson });
  assert.equal(result.status, 'planned');
  assert.equal(result.command, `${JSON.stringify(vestige)} hook session-start`);
  assert.equal(fs.existsSync(path.join(home, '.claude', 'settings.json')), false);
});

test('the stop-memory hook is copied and wired once for each of its events, beside what is already there', () => {
  const home = scratch('loop');
  const settingsPath = path.join(home, '.claude', 'settings.json');
  const mcp = fakeBinary(path.join(home, 'bin'), 'vestige-mcp', '5.0.0');
  const gate = { matcher: '*', hooks: [{ type: 'command', command: 'python3 ~/.operator/gate/operator-gate.py hook --source claude' }] };
  const prompt = { hooks: [{ type: 'command', command: 'echo prompt' }] };
  writeJson(settingsPath, { model: 'opus', hooks: { PreToolUse: [gate], UserPromptSubmit: [prompt] } });

  const first = kernel.wireStopMemoryHook(mcp, { home, writeJson, platform: 'darwin' });
  assert.equal(first.status, 'written');
  assert.deepEqual(first.events, kernel.STOP_MEMORY_EVENTS);
  assert.ok(first.copied);
  const target = kernel.stopMemoryTarget(home);
  assert.equal(fs.readFileSync(target, 'utf8'), fs.readFileSync(kernel.stopMemorySource(), 'utf8'));

  const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
  assert.equal(settings.model, 'opus');
  assert.deepEqual(settings.hooks.PreToolUse, [gate], 'the gate stays first and untouched');
  assert.deepEqual(settings.hooks.UserPromptSubmit[0], prompt);
  const command = `python3 ${target} --mcp ${mcp}`;
  assert.deepEqual(settings.hooks.PostToolUse, [{ matcher: '*', hooks: [{ type: 'command', command, timeout: 30 }] }]);
  assert.deepEqual(settings.hooks.UserPromptSubmit[1], { hooks: [{ type: 'command', command, timeout: 30 }] });
  assert.deepEqual(settings.hooks.Stop, [{ hooks: [{ type: 'command', command, timeout: 30 }] }]);
  assert.deepEqual(settings.hooks.SessionStart, [{ matcher: kernel.RULES_MATCHER, hooks: [{ type: 'command', command, timeout: 30 }] }]);
  assert.equal(kernel.isRulesHook(command), false, 'the rules step must not mistake this hook for its own');
  assert.ok(first.backupPath && fs.existsSync(first.backupPath));

  const second = kernel.wireStopMemoryHook(mcp, { home, writeJson, platform: 'darwin' });
  assert.equal(second.status, 'present');
  assert.equal(second.copied, false, 'the same version is not copied again');
  const again = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
  assert.equal(again.hooks.PostToolUse.length, 1);
  assert.equal(again.hooks.UserPromptSubmit.length, 2);
  assert.equal(again.hooks.Stop.length, 1);
  assert.equal(again.hooks.SessionStart.length, 1);
});

test('an installed stop-memory hook that is newer is kept', () => {
  const home = scratch('loop-newer');
  fs.mkdirSync(path.join(home, '.claude'), { recursive: true });
  const mcp = fakeBinary(path.join(home, 'bin'), 'vestige-mcp', '5.0.0');
  const target = kernel.stopMemoryTarget(home);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, 'VERSION = "99.0.0"\n');

  const result = kernel.wireStopMemoryHook(mcp, { home, writeJson });
  assert.equal(result.status, 'written');
  assert.equal(result.copied, false);
  assert.equal(fs.readFileSync(target, 'utf8'), 'VERSION = "99.0.0"\n');
});

test('a dry run plans the stop-memory hook and writes nothing; Windows uses python and quotes paths', () => {
  const home = scratch('loop-dry');
  fs.mkdirSync(path.join(home, '.claude'), { recursive: true });
  const mcp = path.join(home, 'Program Files', 'vestige-mcp.exe');

  const result = kernel.wireStopMemoryHook(mcp, { home, dryRun: true, writeJson, platform: 'win32' });
  assert.equal(result.status, 'planned');
  assert.deepEqual(result.events, kernel.STOP_MEMORY_EVENTS);
  assert.equal(result.command, `python ${kernel.stopMemoryTarget(home)} --mcp ${JSON.stringify(mcp)}`);
  assert.equal(fs.existsSync(path.join(home, '.claude', 'settings.json')), false);
  assert.equal(fs.existsSync(kernel.stopMemoryTarget(home)), false);
});

test('the stop-memory step leaves a settings file that does not parse exactly as it is', () => {
  const home = scratch('loop-broken');
  const settingsPath = path.join(home, '.claude', 'settings.json');
  fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
  const raw = '{ "hooks": { // mid-edit\n';
  fs.writeFileSync(settingsPath, raw);
  const mcp = fakeBinary(path.join(home, 'bin'), 'vestige-mcp', '5.0.0');

  const result = kernel.wireStopMemoryHook(mcp, { home, writeJson });
  assert.equal(result.status, 'review');
  assert.equal(fs.readFileSync(settingsPath, 'utf8'), raw);
});

test('the hook this package ships is byte-identical to hooks/operator-stop-memory.py in the repo', () => {
  const repoCopy = path.join(__dirname, '..', '..', '..', 'hooks', 'operator-stop-memory.py');
  if (!fs.existsSync(repoCopy)) return;   // the published package has no repo around it
  assert.equal(fs.readFileSync(kernel.stopMemorySource(), 'utf8'), fs.readFileSync(repoCopy, 'utf8'));
});

// A stand-in vestige-mcp that speaks just enough MCP for the hook: no memory carries any tag yet, and
// every smart_ingest is logged and answered with a new id.
const FAKE_MCP = `#!/usr/bin/env python3
import json, sys
log, n = sys.argv[0] + ".log", 0
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    result = {}
    if msg.get("method") == "tools/call":
        name, args = msg["params"]["name"], msg["params"]["arguments"]
        if name == "recall":
            body = {"error": "handle_required"}
        else:
            n += 1
            with open(log, "a") as f:
                f.write(json.dumps(args) + "\\n")
            body = {"results": [{"nodeId": "mem-%016x" % n}]}
        result = {"content": [{"type": "text", "text": json.dumps(body)}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\\n")
    sys.stdout.flush()
`;

test('a gate STOP receipt becomes a rule memory and a stop memory derived_from it, through --mcp', { skip: !POSIX }, () => {
  const { spawnSync } = require('child_process');
  const home = scratch('loop-run');
  const opHome = path.join(home, '.operator');
  fs.mkdirSync(path.join(opHome, 'receipts'), { recursive: true });
  const digest = 'a7892fe4b29ca21b74902d5160da363145b3b82ca8df665756f314a2fc17a91f';
  const row = { decision: 'STOP', digest, prev: '0'.repeat(64), commitment: 'OP-011', rule: 'no-unrestorable-loss',
    ts: new Date().toISOString().slice(0, 19) + 'Z', source: 'claude', session: 's-1', cwd: '/shop', tool: 'Bash',
    action_preview: 'git checkout -- src/app.py', detail: 'discards uncommitted changes', action_digest: 'f9a438193f2525ac6b27f3a9',
    gate_version: '0.3.11' };
  fs.writeFileSync(path.join(opHome, 'receipts', '2026-10-19.jsonl'), JSON.stringify(row) + '\n');
  const mcp = path.join(home, 'fake-mcp');
  fs.writeFileSync(mcp, FAKE_MCP, { mode: 0o755 });

  const payload = JSON.stringify({ hook_event_name: 'PostToolUse', session_id: 's-1' });
  // the scratch home stands in for HOME: a Vestige 6 server running on this machine keeps its reflex
  // heartbeat fresh under the real HOME, and the hook would rightly step aside for it
  const isolated = { ...process.env, HOME: home, XDG_DATA_HOME: '', VESTIGE_DATA_DIR: '', OPERATOR_HOME: opHome };
  const run = spawnSync('python3', [kernel.stopMemorySource(), '--mcp', mcp], {
    input: payload, encoding: 'utf8', timeout: 30000,
    env: { ...isolated, VESTIGE_MCP_BIN: '', PATH: '/usr/bin:/bin' },
  });
  assert.equal(run.status, 0, run.stderr);
  const said = JSON.parse(run.stdout).hookSpecificOutput;
  assert.equal(said.hookEventName, 'PostToolUse');
  assert.match(said.additionalContext, /recall\(handle="op-receipt-a7892fe4b29ca21b"\)/);

  const writes = fs.readFileSync(mcp + '.log', 'utf8').trim().split('\n').map((line) => JSON.parse(line).items[0]);
  assert.equal(writes.length, 2);
  assert.deepEqual(writes[0].tags, ['OP-011', 'op-rule-OP-011', 'operator', 'operator-rule']);
  assert.deepEqual(writes[1].tags, ['OP-011', 'op-receipt-a7892fe4b29ca21b', 'operator', 'operator-stop']);
  assert.deepEqual(writes[1].links, [{ kind: 'derived_from', to: 'mem-0000000000000001' }]);
  assert.match(writes[1].content, /git checkout -- src\/app\.py/);

  const rerun = spawnSync('python3', [kernel.stopMemorySource(), '--mcp', mcp], {
    input: payload, encoding: 'utf8', timeout: 30000, env: isolated,
  });
  assert.equal(rerun.status, 0);
  assert.equal(rerun.stdout, '', 'a stop already recorded is not written or announced again');
  assert.equal(fs.readFileSync(mcp + '.log', 'utf8').trim().split('\n').length, 2);

  // The next session starts and is told what was stopped, with the handle that recalls it.
  const next = spawnSync('python3', [kernel.stopMemorySource(), '--mcp', mcp], {
    input: JSON.stringify({ hook_event_name: 'SessionStart', session_id: 's-2', source: 'startup' }),
    encoding: 'utf8', timeout: 30000, env: isolated,
  });
  assert.equal(next.status, 0, next.stderr);
  const told = JSON.parse(next.stdout).hookSpecificOutput;
  assert.equal(told.hookEventName, 'SessionStart');
  assert.match(told.additionalContext, /Operator stopped 1 action on this machine/);
  assert.match(told.additionalContext, /OP-011 .*git checkout -- src\/app\.py -> recall\(handle="op-receipt-a7892fe4b29ca21b"\)/);
  assert.equal(fs.readFileSync(mcp + '.log', 'utf8').trim().split('\n').length, 2, 'session start writes nothing');
});

test('the gate installer exit codes map to what the summary tells the user', () => {
  assert.equal(kernel.GATE_EXIT[0], 'installed');
  assert.equal(kernel.GATE_EXIT[3], 'agent-shell');
  assert.equal(kernel.GATE_EXIT[4], 'kept-existing');
  assert.equal(kernel.GATE_EXIT[5], 'installed-unwired');
});

test('the data directory follows VESTIGE_DATA_DIR, then the platform default', () => {
  const home = '/home/someone';
  assert.equal(withEnv({ VESTIGE_DATA_DIR: '/srv/vestige' }, () => kernel.dataDir(home, 'linux')), '/srv/vestige');
  assert.equal(withEnv({ VESTIGE_DATA_DIR: undefined }, () => kernel.dataDir(home, 'darwin')),
    path.join(home, 'Library', 'Application Support', 'com.vestige.core'));
  assert.equal(withEnv({ VESTIGE_DATA_DIR: undefined, XDG_DATA_HOME: undefined }, () => kernel.dataDir(home, 'linux')),
    path.join(home, '.local', 'share', 'vestige'));
});
