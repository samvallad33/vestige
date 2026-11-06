'use strict';

// The rest of the system behind `npx @vestige/init`: the binaries, the Strata store, the rules hook
// and the Operator Lite gate. bin/init.js wires the clients and calls these in order.
//
// Every step changes only the user's own machine, keeps what it replaces, and never downgrades:
// a binary or gate that is already newer stays where it is.

const fs = require('fs');
const os = require('os');
const path = require('path');
const readline = require('readline');
const { spawn, spawnSync } = require('child_process');

const BINARIES = ['vestige-mcp', 'vestige', 'vestige-restore', 'vestige-upgrade'];

// Claude Code runs a SessionStart command hook on these sources (hooks/settings.rules.fragment.json).
const RULES_MATCHER = 'startup|resume|clear|compact';

function exe(name, platform = os.platform()) {
  return platform === 'win32' ? `${name}.exe` : name;
}

// The npm package that carries this machine's binaries, pinned in optionalDependencies.
function platformPackage(platform = os.platform(), arch = os.arch()) {
  if (platform === 'darwin' && arch === 'arm64') return '@vestige/mcp-darwin-arm64';
  if (platform === 'darwin' && arch === 'x64') return '@vestige/mcp-darwin-x64';
  if (platform === 'linux' && arch === 'arm64') return '@vestige/mcp-linux-arm64';
  if (platform === 'linux' && arch === 'x64') return '@vestige/mcp-linux-x64';
  if (platform === 'win32' && arch === 'x64') return '@vestige/mcp-win32-x64';
  return null;
}

function bundledBinDir() {
  if (process.env.VESTIGE_BUNDLED_BIN_DIR) return process.env.VESTIGE_BUNDLED_BIN_DIR;
  const pkg = platformPackage();
  if (!pkg) return null;
  try {
    const dir = path.join(path.dirname(require.resolve(`${pkg}/package.json`)), 'bin');
    return fs.existsSync(path.join(dir, exe('vestige-mcp'))) ? dir : null;
  } catch {
    return null;   // npm skips a platform package whose os/cpu do not match, or --omit=optional
  }
}

function installDir(home = os.homedir(), platform = os.platform()) {
  if (process.env.VESTIGE_INSTALL_DIR) return process.env.VESTIGE_INSTALL_DIR;
  return platform === 'win32' ? path.join(home, '.vestige', 'bin') : path.join(home, '.local', 'bin');
}

function parseVersion(text) {
  const match = /(\d+)\.(\d+)\.(\d+)/.exec(text || '');
  return match ? match.slice(1, 4).map(Number) : null;
}

function compareVersions(a, b) {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

function versionOf(binary) {
  if (!binary || !fs.existsSync(binary)) return null;
  const result = spawnSync(binary, ['--version'], { encoding: 'utf8', timeout: 15000 });
  return result.status === 0 ? parseVersion(result.stdout) : null;
}

function onPath(dir) {
  const wanted = path.resolve(dir);
  return (process.env.PATH || '').split(path.delimiter).some((entry) => entry && path.resolve(entry) === wanted);
}

// Copy a file into place by writing beside it and renaming over it. A rename replaces the directory
// entry, so a symlink at the destination is replaced and the file it points at is left untouched.
function placeFile(src, dst) {
  const temp = `${dst}.tmp-${process.pid}-${Date.now()}`;
  fs.copyFileSync(src, temp);
  fs.chmodSync(temp, 0o755);
  try {
    fs.renameSync(temp, dst);
  } catch (error) {
    try { fs.unlinkSync(temp); } catch {}
    throw error;
  }
}

// Step 00: the vestige binaries. Keeps an installed vestige-mcp that is as new as the bundled one;
// otherwise copies the bundled binaries into the install directory.
function ensureBinaries({ found, dryRun = false } = {}) {
  const bundled = bundledBinDir();
  const bundledVersion = bundled ? versionOf(path.join(bundled, exe('vestige-mcp'))) : null;
  const foundVersion = found ? versionOf(found) : null;

  if (found && foundVersion && (!bundledVersion || compareVersions(foundVersion, bundledVersion) >= 0)) {
    return { status: 'kept', binaryPath: found, version: foundVersion.join('.') };
  }
  if (!bundled || !bundledVersion) {
    if (found) return { status: 'kept', binaryPath: found, version: foundVersion ? foundVersion.join('.') : null };
    return {
      status: 'missing',
      reason: `no Vestige binary for ${os.platform()} ${os.arch()} came with this package`,
    };
  }

  const dir = installDir();
  const binaryPath = path.join(dir, exe('vestige-mcp'));
  const sameFile = found && path.resolve(found) === path.resolve(binaryPath);
  const result = {
    status: dryRun ? 'planned' : 'installed',
    binaryPath,
    version: bundledVersion.join('.'),
    dir,
    // An older vestige-mcp in the install directory is replaced; one anywhere else is left alone.
    replaced: sameFile && foundVersion ? foundVersion.join('.') : null,
    olderElsewhere: found && !sameFile ? { path: found, version: foundVersion ? foundVersion.join('.') : null } : null,
    onPath: onPath(dir),
    copied: [],
  };
  if (dryRun) return result;

  fs.mkdirSync(dir, { recursive: true });
  for (const name of BINARIES) {
    const src = path.join(bundled, exe(name));
    if (!fs.existsSync(src)) continue;
    placeFile(src, path.join(dir, exe(name)));
    result.copied.push(exe(name));
  }
  return result;
}

// The store a default vestige-mcp opens: VESTIGE_DATA_DIR, else the platform default
// (crates/vestige-core/src/data_dir.rs). Shown to the user; the server decides for itself.
function dataDir(home = os.homedir(), platform = os.platform()) {
  const fromEnv = process.env.VESTIGE_DATA_DIR;
  if (fromEnv) return fromEnv.startsWith('~') ? path.join(home, fromEnv.slice(1)) : fromEnv;
  if (platform === 'darwin') return path.join(home, 'Library', 'Application Support', 'com.vestige.core');
  if (platform === 'win32') return path.join(process.env.APPDATA || path.join(home, 'AppData', 'Roaming'), 'vestige', 'core', 'data');
  const base = process.env.XDG_DATA_HOME || path.join(home, '.local', 'share');
  const legacy = path.join(base, 'core');
  const preferred = path.join(base, 'vestige');
  const holdsStore = (dir) => fs.existsSync(path.join(dir, 'vestige.db')) ||
    (fs.existsSync(path.join(dir, 'log')) && ['store.meta', 'actor.key', 'receipt-signing.key'].some((n) => fs.existsSync(path.join(dir, n))));
  return !holdsStore(preferred) && holdsStore(legacy) ? legacy : preferred;
}

// Step 01: start vestige-mcp the way a client does (stdio), list its tools, and stop it. A fresh
// machine gets its Strata store here; a machine where another Vestige process holds the store is
// answered through that process.
function startStore(binaryPath, { clientVersion = '0.0.0', timeoutMs = 30000 } = {}) {
  return new Promise((resolve) => {
    let settled = false;
    let serverInfo = null;
    const child = spawn(binaryPath, [], { stdio: ['pipe', 'pipe', 'pipe'] });
    let stderr = '';
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { child.stdin.end(); } catch {}
      const killer = setTimeout(() => { try { child.kill(); } catch {} }, 5000);
      child.once('exit', () => clearTimeout(killer));
      resolve({ ...value, dataDir: dataDir(), storeExists: fs.existsSync(path.join(dataDir(), 'log')) });
    };
    const timer = setTimeout(() => finish({ ok: false, reason: `no answer within ${Math.round(timeoutMs / 1000)} s` }), timeoutMs);
    child.on('error', (error) => finish({ ok: false, reason: error.message }));
    child.on('close', (code) => finish({ ok: false, reason: `vestige-mcp exited with ${code}: ${stderr.trim().split('\n').pop() || 'no message'}` }));
    child.stderr.on('data', (chunk) => { stderr = (stderr + chunk).slice(-4000); });
    const send = (message) => child.stdin.write(JSON.stringify(message) + '\n');
    readline.createInterface({ input: child.stdout }).on('line', (line) => {
      let message;
      try { message = JSON.parse(line); } catch { return; }
      if (message.id === 1) {
        if (message.error) return finish({ ok: false, reason: message.error.message || 'initialize failed' });
        serverInfo = (message.result && message.result.serverInfo) || {};
        send({ jsonrpc: '2.0', method: 'notifications/initialized' });
        send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
      } else if (message.id === 2) {
        if (message.error) return finish({ ok: false, reason: message.error.message || 'tools/list failed' });
        const tools = (message.result && message.result.tools) || [];
        const EXPECTED_TOOLS = ['session', 'remember', 'recall', 'causal_walk', 'curate', 'precedent'];
        const names = tools.map((t) => t.name);
        const missing = EXPECTED_TOOLS.filter((name) => !names.includes(name));
        if (missing.length > 0) {
          return finish({ ok: false, reason: `missing unified tools: ${missing.join(', ')}` });
        }
        finish({ ok: true, server: serverInfo.name || 'vestige', serverVersion: serverInfo.version || null, tools: tools.length });
      }
    });
    send({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: { protocolVersion: '2025-11-25', capabilities: {}, clientInfo: { name: 'vestige-init', version: clientVersion } },
    });
  });
}

function hasClaudeCode(home = os.homedir()) {
  if (fs.existsSync(path.join(home, '.claude'))) return true;
  const which = spawnSync(os.platform() === 'win32' ? 'where' : 'which', ['claude'], { stdio: 'ignore' });
  return which.status === 0;
}

function shellQuote(file) {
  return /^[\w@%+=:,./\\-]+$/.test(file) ? file : JSON.stringify(file);
}

function isRulesHook(command) {
  return typeof command === 'string' && /vestige/.test(command) && /\bhook\s+session-start\b/.test(command);
}

// ~/.claude/settings.json as an object, or a review result. A settings file that does not parse is
// the owner's file mid-edit and is left as it is.
function readSettings(settingsPath) {
  if (!fs.existsSync(settingsPath)) return { settings: {} };
  const raw = fs.readFileSync(settingsPath, 'utf8');
  let settings;
  try {
    settings = raw.trim() ? JSON.parse(raw) : {};
  } catch (error) {
    return { review: { status: 'review', settingsPath, reason: `${settingsPath} is not valid JSON (${error.message}); it was left as it is` } };
  }
  if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
    return { review: { status: 'review', settingsPath, reason: `${settingsPath} does not hold a JSON object; it was left as it is` } };
  }
  return { settings };
}

// Step 04: register `vestige hook session-start` for Claude Code, merged into ~/.claude/settings.json.
// The hook reads only: at every startup, resume, clear and compact it hands the session the decision
// and correction memories tagged `pinned` (and the tags in a repo's .vestige-rules), by exact tag.
function wireRulesHook(vestigeBin, { home = os.homedir(), dryRun = false, writeJson } = {}) {
  if (!hasClaudeCode(home)) return { status: 'skipped', reason: 'Claude Code is not installed here' };
  if (!vestigeBin || (!dryRun && !fs.existsSync(vestigeBin))) return { status: 'skipped', reason: 'the vestige binary was not found beside vestige-mcp' };
  const settingsPath = path.join(home, '.claude', 'settings.json');
  const { settings, review } = readSettings(settingsPath);
  if (review) return review;
  const hooks = settings.hooks && typeof settings.hooks === 'object' ? settings.hooks : {};
  const groups = Array.isArray(hooks.SessionStart) ? hooks.SessionStart : [];
  const present = groups.some((group) => Array.isArray(group && group.hooks) && group.hooks.some((hook) => isRulesHook(hook && hook.command)));
  if (present) return { status: 'present', settingsPath };

  const command = `${shellQuote(vestigeBin)} hook session-start`;
  if (dryRun) return { status: 'planned', settingsPath, command };
  settings.hooks = hooks;
  hooks.SessionStart = groups;
  groups.push({ matcher: RULES_MATCHER, hooks: [{ type: 'command', command, timeout: 10 }] });
  const backupPath = writeJson(settingsPath, settings);
  return { status: 'written', settingsPath, command, backupPath };
}

// The stop-to-memory hook (hooks/operator-stop-memory.py): every Operator stop becomes a Vestige
// memory derived_from a memory for its rule, and the agent is told the exact handle that recalls it.
// It ships in this package and is copied to a fixed place, because npx runs from a cache that does
// not last.
const STOP_MEMORY_EVENTS = ['PostToolUse', 'UserPromptSubmit', 'Stop', 'SessionStart'];
const STOP_MEMORY_MATCHERS = { PostToolUse: '*', SessionStart: RULES_MATCHER };

function stopMemorySource() {
  return path.join(__dirname, '..', 'hooks', 'operator-stop-memory.py');
}

function stopMemoryTarget(home = os.homedir()) {
  return path.join(home, '.vestige', 'hooks', 'operator-stop-memory.py');
}

function hookVersion(file) {
  try {
    const match = /^VERSION = "(\d+\.\d+\.\d+)"/m.exec(fs.readFileSync(file, 'utf8'));
    return match ? parseVersion(match[1]) : null;
  } catch {
    return null;
  }
}

function isStopMemoryHook(command) {
  return typeof command === 'string' && /operator-stop-memory\.py/.test(command);
}

function stopMemoryCommand(hookPath, mcpBin, platform = os.platform()) {
  return `${platform === 'win32' ? 'python' : 'python3'} ${shellQuote(hookPath)} --mcp ${shellQuote(mcpBin)}`;
}

// Step 05, second half: once a gate is in place, its stops become memory. Copies the hook (an
// installed copy as new as this one is kept) and registers it in ~/.claude/settings.json, once per
// event, beside every hook already there: PostToolUse, UserPromptSubmit and Stop record new stops,
// and SessionStart hands each new session the recent ones with the handles that recall them.
function wireStopMemoryHook(mcpBin, { home = os.homedir(), dryRun = false, writeJson, source = stopMemorySource(), platform = os.platform() } = {}) {
  if (!hasClaudeCode(home)) return { status: 'skipped', reason: 'Claude Code is not installed here' };
  if (!mcpBin || (!dryRun && !fs.existsSync(mcpBin))) return { status: 'skipped', reason: 'vestige-mcp was not found' };
  if (!fs.existsSync(source)) return { status: 'skipped', reason: 'the stop-memory hook did not come with this package' };
  const settingsPath = path.join(home, '.claude', 'settings.json');
  const { settings, review } = readSettings(settingsPath);
  if (review) return review;
  const hooks = settings.hooks && typeof settings.hooks === 'object' ? settings.hooks : {};
  const wired = (event) => Array.isArray(hooks[event]) && hooks[event].some((group) =>
    Array.isArray(group && group.hooks) && group.hooks.some((hook) => isStopMemoryHook(hook && hook.command)));
  const events = STOP_MEMORY_EVENTS.filter((event) => !wired(event));

  const target = stopMemoryTarget(home);
  const command = stopMemoryCommand(target, mcpBin, platform);
  const sourceVersion = hookVersion(source);
  const targetVersion = hookVersion(target);
  const copy = !targetVersion || Boolean(sourceVersion && compareVersions(sourceVersion, targetVersion) > 0);
  if (dryRun) return { status: 'planned', settingsPath, command, target, events, copy };

  if (copy) {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    placeFile(source, target);
  }
  if (events.length === 0) return { status: 'present', settingsPath, target, copied: copy };
  settings.hooks = hooks;
  for (const event of events) {
    const groups = Array.isArray(hooks[event]) ? hooks[event] : [];
    const entry = { type: 'command', command, timeout: 30 };
    groups.push(STOP_MEMORY_MATCHERS[event] ? { matcher: STOP_MEMORY_MATCHERS[event], hooks: [entry] } : { hooks: [entry] });
    hooks[event] = groups;
  }
  const backupPath = writeJson(settingsPath, settings);
  return { status: 'written', settingsPath, command, target, events, copied: copy, backupPath };
}

function gateLauncher() {
  try {
    return path.join(path.dirname(require.resolve('operator-lite/package.json')), 'bin', 'operator-lite.js');
  } catch {
    return null;
  }
}

function gateVersion() {
  try {
    return require('operator-lite/package.json').version;
  } catch {
    return null;
  }
}

// operator-gate.py cmd_install returns these; operator-lite.js returns 1 when no Python 3.9+ is found.
const GATE_EXIT = {
  0: 'installed',
  1: 'failed',
  3: 'agent-shell',
  4: 'kept-existing',
  5: 'installed-unwired',
};

function runGate(args, { onLine = () => {} } = {}) {
  const launcher = gateLauncher();
  if (!launcher) return Promise.resolve({ code: null, output: '', missing: true });
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [launcher, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '';
    const take = (stream) => readline.createInterface({ input: stream }).on('line', (line) => {
      output += line + '\n';
      onLine(line);
    });
    take(child.stdout);
    take(child.stderr);
    child.on('error', (error) => resolve({ code: null, output: output + error.message }));
    child.on('close', (code) => resolve({ code, output }));
  });
}

// Step 05: install Operator Lite, the free gate, and wire every agent host it finds. The gate's own
// installer decides: it refuses in an agent's shell (3) and never replaces the paid or a newer gate (4).
async function installGate({ dryRun = false, onLine } = {}) {
  const version = gateVersion();
  if (!gateLauncher()) return { status: 'missing', reason: 'operator-lite did not come with this package' };
  if (dryRun) return { status: 'planned', version };
  const { code, output } = await runGate([], { onLine });
  return { status: GATE_EXIT[code] || 'failed', code, version, output };
}

// Step 06: the gate replays the GuardFall bypass corpus it carries. It classifies every command and
// runs none.
async function gateSelfTest() {
  const { code, output } = await runGate(['test']);
  const match = /corpus=(\S+) cases=(\d+) passed=(\d+) failed=(\d+)/.exec(output || '');
  if (!match) return { ok: false, code, reason: (output || '').trim().split('\n').pop() || 'no result' };
  const [, corpus, cases, passed, failed] = match;
  return { ok: code === 0 && Number(failed) === 0, corpus, cases: Number(cases), passed: Number(passed), failed: Number(failed) };
}

module.exports = {
  BINARIES,
  RULES_MATCHER,
  GATE_EXIT,
  exe,
  platformPackage,
  bundledBinDir,
  installDir,
  parseVersion,
  compareVersions,
  placeFile,
  ensureBinaries,
  dataDir,
  startStore,
  hasClaudeCode,
  isRulesHook,
  wireRulesHook,
  STOP_MEMORY_EVENTS,
  stopMemorySource,
  stopMemoryTarget,
  stopMemoryCommand,
  isStopMemoryHook,
  wireStopMemoryHook,
  gateLauncher,
  gateVersion,
  installGate,
  gateSelfTest,
};
