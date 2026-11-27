'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { spawnSync } = require('node:child_process');

const PKG_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PKG_ROOT, '..', '..');
const RUNTIMES_DIR = path.join(REPO_ROOT, 'hooks', 'runtimes');
const BRIDGE_JS = path.join(PKG_ROOT, 'hooks', 'vestige-bridge.cjs');
const HOOKS_DIR = path.join(PKG_ROOT, 'hooks');
const LIB_DIR = path.join(PKG_ROOT, 'lib');
const SETUP_LIB = path.join(LIB_DIR, 'vestige-memory-setup.js');

const { RUNTIMES, installRuntime } = require(path.join(RUNTIMES_DIR, 'install.js'));

const HOME = process.env.USERPROFILE || process.env.HOME || os.homedir();

function hasUtf8Bom(buf) {
  return buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf;
}

function findExe(names, envVar) {
  if (process.env[envVar] && fs.existsSync(process.env[envVar])) return process.env[envVar];
  const roots = [
    path.join('C:\\cargo-target-v71', 'release'),
    path.join('C:\\cargo-target-v71', 'debug'),
    path.join(REPO_ROOT, 'target', 'release'),
    path.join(REPO_ROOT, 'target', 'debug'),
    path.join(HOME, '.vestige', 'bin'),
  ];
  for (const dir of roots) {
    for (const name of names) {
      const c = path.join(dir, name);
      if (fs.existsSync(c)) return c;
    }
  }
  return null;
}

function findPython() {
  for (const cmd of [process.env.PYTHON, 'python3', 'python']) {
    if (!cmd) continue;
    const r = spawnSync(cmd, ['--version'], { encoding: 'utf8', windowsHide: true, timeout: 3000 });
    if (r.status === 0 && /Python 3\./.test((r.stdout || '') + (r.stderr || ''))) return cmd;
  }
  return 'python3';
}

const VESTIGE_MCP_BIN = findExe(process.platform === 'win32' ? ['vestige-mcp.exe'] : ['vestige-mcp'], 'VESTIGE_MCP_BIN');
const VESTIGE_BIN = findExe(process.platform === 'win32' ? ['vestige.exe'] : ['vestige'], 'VESTIGE_BIN');

function createTempBridgeEnv({ requireGate = true } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-multi-rt-'));
  const dataDir = path.join(root, 'store');
  const opHome = path.join(root, 'operator');
  const gateDir = path.join(opHome, 'gate');
  const memDir = path.join(root, 'memory');
  const hooksCopyDir = path.join(memDir, 'hooks');
  const libCopyDir = path.join(memDir, 'lib');
  for (const d of [dataDir, gateDir, hooksCopyDir, libCopyDir]) fs.mkdirSync(d, { recursive: true });

  fs.writeFileSync(path.join(opHome, 'mode'), 'enforce\n');
  for (const f of ['operator-gate.py', 'lite_core.py', 'session_automaton.json']) {
    fs.copyFileSync(path.join(HOOKS_DIR, f), path.join(gateDir, f));
  }
  for (const f of ['vestige-bridge.cjs', 'agy-bridge.cjs', 'memory.cjs', 'reflex.cjs', 'guard.cjs', 'stop-memory.cjs']) {
    fs.copyFileSync(path.join(HOOKS_DIR, f), path.join(hooksCopyDir, f));
  }
  fs.copyFileSync(path.join(LIB_DIR, 'memory-triggers.cjs'), path.join(libCopyDir, 'memory-triggers.cjs'));

  const triggersJson = path.join(memDir, 'triggers.json');
  const { defaults } = require(path.join(LIB_DIR, 'memory-triggers.cjs'));
  fs.writeFileSync(triggersJson, JSON.stringify(defaults, null, 2));

  const memCfgPath = path.join(memDir, 'config.json');
  fs.writeFileSync(memCfgPath, JSON.stringify({
    version: '7.0.0',
    binary: VESTIGE_MCP_BIN,
    binaryArgs: ['--data-dir', dataDir],
    triggers: triggersJson,
    deadlineMs: 12000,
  }, null, 2));

  const bridgeCfgPath = path.join(memDir, 'bridge.json');
  fs.writeFileSync(bridgeCfgPath, JSON.stringify({
    python: findPython(),
    gate: requireGate ? path.join(gateDir, 'operator-gate.py') : '',
    requireGate,
    vestigeBin: VESTIGE_BIN || VESTIGE_MCP_BIN,
    dataDir,
    memoryConfig: memCfgPath,
    hooksDir: hooksCopyDir,
    gateTimeoutMs: 15000,
    memoryDeadlineMs: 12000,
    reflexDeadlineMs: 10000,
    capsuleTimeoutMs: 8000,
  }, null, 2));

  const callBridge = (runtime, event, payload, extraEnv = {}) => {
    const r = spawnSync(
      process.execPath,
      [path.join(hooksCopyDir, 'vestige-bridge.cjs'), '--runtime', runtime, '--event', event, '--config', bridgeCfgPath],
      {
        cwd: root,
        env: {
          ...process.env,
          HOME: root,
          USERPROFILE: root,
          VESTIGE_DATA_DIR: dataDir,
          OPERATOR_HOME: opHome,
          VESTIGE_HOOK_DEADLINE_MS: '12000',
          ...extraEnv,
        },
        input: JSON.stringify(payload),
        encoding: 'utf8',
        timeout: 25000,
        windowsHide: true,
      }
    );
    let parsed = null;
    try { parsed = r.stdout && r.stdout.trim() ? JSON.parse(r.stdout.trim()) : null; } catch {}
    return { status: r.status, stdout: r.stdout, stderr: r.stderr, json: parsed };
  };

  return {
    root,
    dataDir,
    opHome,
    bridgeCfgPath,
    bridgeScript: path.join(hooksCopyDir, 'vestige-bridge.cjs'),
    callBridge,
    cleanup: () => fs.rmSync(root, { recursive: true, force: true }),
  };
}

test('1. All 24 dedicated runtime folders exist under hooks/runtimes/ with BOM-free schemas and zero private user paths', () => {
  assert.ok(fs.existsSync(path.join(RUNTIMES_DIR, 'README.md')), 'hooks/runtimes/README.md missing');
  assert.equal(RUNTIMES.length, 24, 'Expected 24 documented runtimes');

  const PRIVATE_PATH_RE = /(?:[A-Za-z]:\\Users\\|\/Users\/|\/home\/)[A-Za-z0-9._-]+/;
  for (const rt of RUNTIMES) {
    const dir = path.join(RUNTIMES_DIR, rt.id);
    assert.ok(fs.existsSync(dir), `Missing runtime folder hooks/runtimes/${rt.id}`);
    assert.ok(fs.existsSync(path.join(dir, 'README.md')), `Missing README.md for ${rt.id}`);
    const cfgPath = path.join(dir, rt.configFileName);
    assert.ok(fs.existsSync(cfgPath), `Missing config/adapter ${rt.configFileName} in ${rt.id}`);

    const filesToCheck = [];
    const walk = (d) => {
      for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
        if (entry.name === '__pycache__' || entry.name.endsWith('.pyc')) continue;
        const full = path.join(d, entry.name);
        if (entry.isDirectory()) walk(full);
        else filesToCheck.push(full);
      }
    };
    walk(dir);

    for (const file of filesToCheck) {
      const rawBuf = fs.readFileSync(file);
      assert.equal(hasUtf8Bom(rawBuf), false, `${file} must NOT have UTF-8 BOM`);
      const text = rawBuf.toString('utf8');
      assert.equal(PRIVATE_PATH_RE.test(text), false, `${file} leaks private home path`);
      if (file.endsWith('.json')) {
        assert.doesNotThrow(() => JSON.parse(text), `${file} must be valid JSON`);
      }
    }
  }

  // Exact upstream 2026 doc-verified checks
  const codex = JSON.parse(fs.readFileSync(path.join(RUNTIMES_DIR, 'codex', 'hooks.json'), 'utf8'));
  for (const ev of ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PermissionRequest', 'PostToolUse', 'PreCompact', 'PostCompact', 'Stop', 'SubagentStart', 'SubagentStop', 'Interrupt', 'SessionEnd']) {
    assert.ok(Array.isArray(codex.hooks[ev]) && codex.hooks[ev].length > 0, `codex/hooks.json missing ${ev}`);
  }

  const cursor = JSON.parse(fs.readFileSync(path.join(RUNTIMES_DIR, 'cursor', 'hooks.json'), 'utf8'));
  assert.equal(cursor.version, 1);
  assert.equal(cursor.hooks.preToolUse[0].failClosed, true);
  assert.equal(cursor.hooks.beforeShellExecution[0].failClosed, true);
  assert.equal(cursor.hooks.beforeMCPExecution[0].failClosed, true);
  assert.equal(cursor.hooks.stop[0].failClosed, true);
  assert.ok(cursor.hooks.beforeSubmitPrompt.length > 0);

  const gemini = JSON.parse(fs.readFileSync(path.join(RUNTIMES_DIR, 'gemini-cli', 'settings.json'), 'utf8'));
  assert.equal(gemini.hooks.BeforeTool[0].hooks[0].timeout, 15000);
  assert.ok(fs.existsSync(path.join(RUNTIMES_DIR, 'gemini-cli', 'gemini-extension.json')));
  assert.ok(fs.existsSync(path.join(RUNTIMES_DIR, 'gemini-cli', 'hooks', 'hooks.json')));

  const copilot = JSON.parse(fs.readFileSync(path.join(RUNTIMES_DIR, 'copilot', 'vestige.json'), 'utf8'));
  assert.equal(copilot.version, 1);
  assert.ok(copilot.hooks.preToolUse[0].bash && copilot.hooks.preToolUse[0].powershell);
  assert.equal(copilot.hooks.preToolUse[0].timeoutSec, 15);

  const kiro = JSON.parse(fs.readFileSync(path.join(RUNTIMES_DIR, 'kiro', 'vestige.json'), 'utf8'));
  assert.equal(kiro.version, 'v1');
  assert.ok(kiro.hooks.some(h => h.trigger === 'PreToolUse') && kiro.hooks.some(h => h.trigger === 'PostFileSave'));
});

test('2. Multi-runtime dialect normalization + Call Coach CC-001..CC-013 across Codex, Cursor, Gemini, Copilot, Cline, Windsurf, OpenCode', () => {
  const env = createTempBridgeEnv({ requireGate: false });
  try {
    // Codex PreToolUse rewrite omits continue/stopReason and sets updatedInput + permissionDecision: allow
    const codexRewrite = env.callBridge('codex', 'PreToolUse', {
      session_id: 's-codex',
      cwd: env.root,
      tool_name: 'mcp__vestige__recall',
      tool_input: { id: 'mem-1234567890abcdef' },
    });
    assert.equal(codexRewrite.status, 0);
    assert.equal(codexRewrite.json.hookSpecificOutput.hookEventName, 'PreToolUse');
    assert.equal(codexRewrite.json.hookSpecificOutput.permissionDecision, 'allow');
    assert.equal(codexRewrite.json.hookSpecificOutput.updatedInput.handle, 'mem-1234567890abcdef');
    assert.equal(codexRewrite.json.continue, undefined);
    assert.equal(codexRewrite.json.stopReason, undefined);

    // Codex CC-010 deny
    const codexDeny = env.callBridge('codex', 'PreToolUse', {
      session_id: 's-codex',
      cwd: env.root,
      tool_name: 'mcp__vestige__curate',
      tool_input: { action: 'delete', memoryId: 'mem-1234567890abcdef' },
    });
    assert.equal(codexDeny.status, 0);
    assert.equal(codexDeny.json.hookSpecificOutput.permissionDecision, 'deny');
    assert.match(codexDeny.json.hookSpecificOutput.permissionDecisionReason, /CC-010/);

    // Gemini BeforeTool rewrite uses hookSpecificOutput.tool_input
    const gemRewrite = env.callBridge('gemini-cli', 'BeforeTool', {
      session_id: 's-gem',
      cwd: env.root,
      tool_name: 'mcp_vestige_remember',
      tool_input: { text: 'Saved architectural invariant' },
    });
    assert.equal(gemRewrite.status, 0);
    assert.equal(gemRewrite.json.hookSpecificOutput.hookEventName, 'BeforeTool');
    assert.equal(gemRewrite.json.hookSpecificOutput.tool_input.content, 'Saved architectural invariant');

    // Cursor beforeMCPExecution parses stringified tool_input and returns updated_input
    const curRewrite = env.callBridge('cursor', 'beforeMCPExecution', {
      conversation_id: 'c-cur',
      workspace_roots: [env.root],
      mcp_server_name: 'vestige',
      tool_name: 'causal_walk',
      tool_input: '{"handle":"mem-1234567890abcdef"}',
    });
    assert.equal(curRewrite.status, 0);
    assert.equal(curRewrite.json.permission, 'allow');
    assert.equal(curRewrite.json.updated_input.from_handle, 'mem-1234567890abcdef');
    assert.equal(curRewrite.json.updated_input.scan_limit, 500);

    // Cursor beforeSubmitPrompt always returns continue: true
    const curPrompt = env.callBridge('cursor', 'beforeSubmitPrompt', {
      conversation_id: 'c-cur',
      workspace_roots: [env.root],
      prompt: 'Check auth',
    });
    assert.equal(curPrompt.status, 0);
    assert.equal(curPrompt.json.continue, true);

    // Cline PreToolUse CC-010 returns { cancel: true, errorMessage } at exit 0
    const clineDeny = env.callBridge('cline', 'PreToolUse', {
      taskId: 'cline-1',
      workspaceRoots: [env.root],
      tool_call: {
        name: 'use_mcp_tool',
        input: { server_name: 'vestige', tool_name: 'memory', arguments: '{"action":"purge"}' },
      },
    });
    assert.equal(clineDeny.status, 0);
    assert.equal(clineDeny.json.cancel, true);
    assert.match(clineDeny.json.errorMessage, /CC-010/);

    // Copilot preToolUse CC-010 returns { permissionDecision: 'deny' }
    const copilotDeny = env.callBridge('copilot', 'preToolUse', {
      sessionId: 'cop-1',
      cwd: env.root,
      toolName: 'mcp__vestige__curate',
      toolArgs: JSON.stringify({ action: 'delete' }),
    });
    assert.equal(copilotDeny.status, 0);
    assert.equal(copilotDeny.json.permissionDecision, 'deny');
  } finally {
    env.cleanup();
  }
});

test('3. Fail-closed Operator Gate + mandatory precedent(query="stops") verified in EVERY host dialect', () => {
  const env = createTempBridgeEnv({ requireGate: true });
  try {
    // Codex PreToolUse -> exit 0 + hookSpecificOutput.permissionDecision: "deny" (OP-004)
    const codexBlock = env.callBridge('codex', 'PreToolUse', {
      session_id: 'op-codex',
      cwd: env.root,
      tool_name: 'Bash',
      tool_input: { command: 'git push --force origin main' },
    });
    assert.equal(codexBlock.status, 0);
    assert.equal(codexBlock.json.hookSpecificOutput.permissionDecision, 'deny');
    assert.match(codexBlock.json.hookSpecificOutput.permissionDecisionReason, /OP-004/);
    // Subsequent mutation is blocked until precedent(query="stops")
    const codexFollowup = env.callBridge('codex', 'PreToolUse', {
      session_id: 'op-codex',
      cwd: env.root,
      tool_name: 'Bash',
      tool_input: { command: 'git status' },
    });
    assert.equal(codexFollowup.json.hookSpecificOutput.permissionDecision, 'deny');
    assert.match(codexFollowup.json.hookSpecificOutput.permissionDecisionReason, /precedent\(query="stops"\)/);

    // Cursor beforeShellExecution -> exit 0 + { permission: "deny", agent_message, user_message }
    const cursorBlock = env.callBridge('cursor', 'beforeShellExecution', {
      conversation_id: 'op-cursor',
      cwd: env.root,
      command: 'git push --force origin main',
    });
    assert.equal(cursorBlock.status, 0);
    assert.equal(cursorBlock.json.permission, 'deny');
    assert.match(cursorBlock.json.agent_message, /OP-004/);

    // Gemini CLI BeforeTool -> exit 0 + { decision: "deny", reason }
    const gemBlock = env.callBridge('gemini-cli', 'BeforeTool', {
      session_id: 'op-gemini',
      cwd: env.root,
      tool_name: 'run_shell_command',
      tool_input: { command: 'git push --force origin main' },
    });
    assert.equal(gemBlock.status, 0);
    assert.equal(gemBlock.json.decision, 'deny');
    assert.match(gemBlock.json.reason, /OP-004/);

    // Windsurf pre_run_command -> exit 2 + stderr!
    const wsBlock = env.callBridge('windsurf', 'pre_run_command', {
      trajectory_id: 'op-ws',
      tool_info: { command_line: 'git push --force origin main', cwd: env.root },
    });
    assert.equal(wsBlock.status, 2);
    assert.match(wsBlock.stderr, /OP-004/);

    // Kiro PreToolUse -> exit 2 + stderr!
    const kiroBlock = env.callBridge('kiro', 'PreToolUse', {
      session_id: 'op-kiro',
      cwd: env.root,
      tool_name: 'Bash',
      tool_input: { command: 'git push --force origin main' },
    });
    assert.equal(kiroBlock.status, 2);
    assert.match(kiroBlock.stderr, /OP-004/);

    // Cline PreToolUse -> exit 0 + { cancel: true }
    const clineBlock = env.callBridge('cline', 'PreToolUse', {
      taskId: 'op-cline',
      workspaceRoots: [env.root],
      tool_call: { name: 'run_commands', input: { command: 'git push --force origin main' } },
    });
    assert.equal(clineBlock.status, 0);
    assert.equal(clineBlock.json.cancel, true);
    assert.match(clineBlock.json.errorMessage, /OP-004/);
  } finally {
    env.cleanup();
  }
});

test('4. OpenCode JS plugin enforces tool.execute.before block & Call Coach arg mutation', async () => {
  const env = createTempBridgeEnv({ requireGate: true });
  const prevBridge = process.env.VESTIGE_BRIDGE_PATH;
  const prevCfg = process.env.VESTIGE_BRIDGE_CONFIG;
  const prevOp = process.env.OPERATOR_HOME;
  const prevData = process.env.VESTIGE_DATA_DIR;
  try {
    process.env.VESTIGE_BRIDGE_PATH = env.bridgeScript;
    process.env.VESTIGE_BRIDGE_CONFIG = env.bridgeCfgPath;
    process.env.OPERATOR_HOME = env.opHome;
    process.env.VESTIGE_DATA_DIR = env.dataDir;

    const mod = await import(pathToFileURL(path.join(RUNTIMES_DIR, 'opencode', 'index.js')).href);
    const plugin = await mod.VestigePlugin({ directory: env.root, worktree: env.root });

    const coachOut = { args: { id: 'mem-1234567890abcdef' } };
    await plugin['tool.execute.before']({ tool: 'mcp__vestige__recall', sessionID: 'oc-1', callID: 'c1' }, coachOut);
    assert.equal(coachOut.args.handle, 'mem-1234567890abcdef');

    await assert.rejects(
      plugin['tool.execute.before']({ tool: 'bash', sessionID: 'oc-1', callID: 'c2' }, { args: { command: 'git push --force origin main' } }),
      /OP-004/
    );
  } finally {
    if (prevBridge === undefined) delete process.env.VESTIGE_BRIDGE_PATH; else process.env.VESTIGE_BRIDGE_PATH = prevBridge;
    if (prevCfg === undefined) delete process.env.VESTIGE_BRIDGE_CONFIG; else process.env.VESTIGE_BRIDGE_CONFIG = prevCfg;
    if (prevOp === undefined) delete process.env.OPERATOR_HOME; else process.env.OPERATOR_HOME = prevOp;
    if (prevData === undefined) delete process.env.VESTIGE_DATA_DIR; else process.env.VESTIGE_DATA_DIR = prevData;
    env.cleanup();
  }
});

test('5. Live vestige-mcp Zero-Prompt Recall & Prompt Memory Injection across Codex, Cursor & Gemini', { skip: !VESTIGE_MCP_BIN }, () => {
  const env = createTempBridgeEnv({ requireGate: false });
  try {
    const reqs = [
      JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'multi-e2e', version: '7.0.0' } } }),
      JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
      JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'remember', arguments: { action: 'save', node_type: 'fact', content: 'Standalone deployment target uses baremetal Kubernetes workers with WireGuard mesh.', tags: ['deploy', 'pinned'] } } }),
    ].join('\n') + '\n';
    const seed = spawnSync(VESTIGE_MCP_BIN, ['--data-dir', env.dataDir], { input: reqs, encoding: 'utf8', timeout: 15000, windowsHide: true });
    assert.equal(seed.status, 0, seed.stderr);
    const seedAns = seed.stdout.trim().split(/\r?\n/).map(l => { try { return JSON.parse(l); } catch { return null; } }).find(m => m && m.id === 2);
    assert.ok(seedAns && seedAns.result && !seedAns.result.isError, `Seed remember failed: ${seed.stdout}`);

    const codexStart = env.callBridge('codex', 'SessionStart', { session_id: 'e2e-codex', cwd: env.root, source: 'startup' });
    assert.equal(codexStart.status, 0);
    assert.ok(codexStart.json.hookSpecificOutput && typeof codexStart.json.hookSpecificOutput.additionalContext === 'string', `codexStart missing context: ${codexStart.stdout}`);
    assert.match(codexStart.json.hookSpecificOutput.additionalContext, /WireGuard|Rulebook|baremetal/);

    const codexPrompt = env.callBridge('codex', 'UserPromptSubmit', { session_id: 'e2e-codex-prompt', cwd: env.root, prompt: 'How is our baremetal Kubernetes WireGuard deployment configured?' });
    assert.equal(codexPrompt.status, 0);
    assert.ok(codexPrompt.json.hookSpecificOutput && typeof codexPrompt.json.hookSpecificOutput.additionalContext === 'string', `codexPrompt missing context: ${codexPrompt.stdout}`);
    assert.match(codexPrompt.json.hookSpecificOutput.additionalContext, /WireGuard/);

    const gemPrompt = env.callBridge('gemini-cli', 'BeforeAgent', { session_id: 'e2e-gem', cwd: env.root, prompt: 'How is our baremetal Kubernetes WireGuard deployment configured?' });
    assert.equal(gemPrompt.status, 0);
    assert.ok(gemPrompt.json.hookSpecificOutput && typeof gemPrompt.json.hookSpecificOutput.additionalContext === 'string', `gemPrompt missing context: ${gemPrompt.stdout}`);
    assert.match(gemPrompt.json.hookSpecificOutput.additionalContext, /WireGuard/);

    const cursorStart = env.callBridge('cursor', 'sessionStart', { conversation_id: 'e2e-cur', cwd: env.root });
    assert.equal(cursorStart.status, 0);
    assert.ok(typeof cursorStart.json.additional_context === 'string' && cursorStart.json.additional_context.length > 0, `cursorStart empty: ${cursorStart.stdout}`);
  } finally {
    env.cleanup();
  }
});

test('6. install.js & @vestige/init install & uninstall multi-runtime hooks idempotently across repo & home scopes', () => {
  const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-rt-install-'));
  const tmpRepo = path.join(tmpHome, 'sample-repo');
  const fakeBin = path.join(tmpHome, 'vestige-mcp.cjs');
  fs.mkdirSync(tmpRepo, { recursive: true });
  fs.writeFileSync(fakeBin, "if (process.argv.includes('--version')) { console.log('vestige-mcp 7.0.0'); process.exit(0); }\n", { mode: 0o755 });

  try {
    // 1) Repo-scoped drop-ins via install.js
    for (const id of ['copilot', 'codex', 'cursor', 'kiro', 'opencode', 'cline']) {
      const res = installRuntime(id, { home: tmpHome, projectDir: tmpRepo, guard: true });
      assert.equal(res.status, 'installed', `${id} install failed`);
      assert.ok(res.written.length > 0 && res.written.every(f => fs.existsSync(f)));
    }
    assert.ok(fs.existsSync(path.join(tmpRepo, '.github', 'hooks', 'vestige.json')));
    assert.ok(fs.existsSync(path.join(tmpRepo, '.codex', 'hooks.json')));
    assert.ok(fs.existsSync(path.join(tmpRepo, '.cursor', 'hooks.json')));
    assert.ok(fs.existsSync(path.join(tmpRepo, '.kiro', 'hooks', 'vestige.json')));
    assert.ok(fs.existsSync(path.join(tmpRepo, '.opencode', 'plugins', 'vestige', 'index.js')));
    assert.ok(fs.existsSync(path.join(tmpRepo, '.clinerules', 'hooks', 'PreToolUse')));

    // 2) @vestige/init setup integration across new global runtimes
    const setup = require(SETUP_LIB);
    fs.mkdirSync(path.join(tmpHome, '.codeium', 'windsurf'), { recursive: true });
    fs.mkdirSync(path.join(tmpHome, '.factory'), { recursive: true });
    fs.mkdirSync(path.join(tmpHome, '.kiro'), { recursive: true });
    fs.mkdirSync(path.join(tmpHome, '.cline'), { recursive: true });

    const p = setup.plan({ home: tmpHome, platform: process.platform, env: { PATH: '', VESTIGE_MCP_PATH: fakeBin }, clients: ['windsurf', 'factory-droid', 'kiro', 'cline'] });
    assert.equal(p.detected.map(c => c.id).sort().join(','), 'cline,factory-droid,kiro,windsurf');
    const applied = setup.apply(p);
    assert.ok(applied.written >= 8);

    assert.ok(JSON.parse(fs.readFileSync(path.join(tmpHome, '.codeium', 'windsurf', 'hooks.json'), 'utf8')).hooks.pre_run_command.length > 0);
    assert.ok(JSON.parse(fs.readFileSync(path.join(tmpHome, '.factory', 'hooks.json'), 'utf8')).hooks.PreToolUse.length > 0);
    assert.equal(JSON.parse(fs.readFileSync(path.join(tmpHome, '.kiro', 'hooks', 'vestige.json'), 'utf8')).version, 'v1');
    assert.ok(fs.existsSync(path.join(tmpHome, '.cline', 'hooks', 'PreToolUse')));
    assert.ok(fs.existsSync(path.join(tmpHome, '.vestige', 'memory', 'hooks', 'vestige-bridge.cjs')));

    // Repeat plan() is idempotent
    const again = setup.plan({ home: tmpHome, platform: process.platform, env: { PATH: '', VESTIGE_MCP_PATH: fakeBin }, clients: ['windsurf', 'factory-droid', 'kiro', 'cline'] });
    assert.equal(again.changes.length, 0, 'Repeat setup must be idempotent');

    // Clean uninstall restores
    const uninst = setup.uninstall(tmpHome);
    assert.equal(uninst.kept.length, 0);
    assert.equal(fs.existsSync(path.join(tmpHome, '.cline', 'hooks', 'PreToolUse')), false);
  } finally {
    fs.rmSync(tmpHome, { recursive: true, force: true });
  }
});
