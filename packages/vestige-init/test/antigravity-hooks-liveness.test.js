'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const PKG_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PKG_ROOT, '..', '..');
const HOOKS_DIR = path.join(PKG_ROOT, 'hooks');
const LIB_DIR = path.join(PKG_ROOT, 'lib');
const SETUP_LIB = path.join(LIB_DIR, 'vestige-memory-setup.js');

const HOME = process.env.USERPROFILE || process.env.HOME || os.homedir();
const HAS_AGY_ENV = fs.existsSync(path.join(HOME, '.gemini', 'config'));
const AGY_PLUGIN_DIR = path.join(HOME, '.gemini', 'config', 'plugins', 'vestige');
const AGY_HOOKS_JSON = path.join(AGY_PLUGIN_DIR, 'hooks.json');
const AGY_BRIDGE_JS = path.join(AGY_PLUGIN_DIR, 'bridge', 'agy-bridge.cjs');
const SHIPPED_BRIDGE_JS = path.join(HOOKS_DIR, 'agy-bridge.cjs');
const SHIPPED_HOOKS_JSON = path.join(HOOKS_DIR, 'agy-hooks.json');

function hasUtf8Bom(buf) {
  return buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf;
}

function findExe(names, envVar) {
  if (process.env[envVar] && fs.existsSync(process.env[envVar])) {
    return process.env[envVar];
  }
  const roots = [
    path.join('C:\\cargo-target-v71', 'release'),
    path.join('C:\\cargo-target-v71', 'debug'),
    path.join(REPO_ROOT, 'target', 'release'),
    path.join(REPO_ROOT, 'target', 'debug'),
    path.join(HOME, '.vestige', 'bin'),
  ];
  for (const dir of roots) {
    for (const name of names) {
      const candidate = path.join(dir, name);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  return null;
}

function findPython() {
  for (const cmd of [process.env.PYTHON, 'python3', 'python']) {
    if (!cmd) continue;
    const r = spawnSync(cmd, ['--version'], { encoding: 'utf8', windowsHide: true, timeout: 3000 });
    if (r.status === 0 && /Python 3\./.test((r.stdout || '') + (r.stderr || ''))) {
      return cmd;
    }
  }
  return 'python3';
}

const VESTIGE_MCP_BIN = findExe(
  process.platform === 'win32' ? ['vestige-mcp.exe'] : ['vestige-mcp'],
  'VESTIGE_MCP_BIN'
);
const VESTIGE_BIN = findExe(
  process.platform === 'win32' ? ['vestige.exe'] : ['vestige'],
  'VESTIGE_BIN'
);

function createIsolatedBridgeEnv() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-e2e-v7-'));
  const dataDir = path.join(root, 'store');
  const opHome = path.join(root, 'operator');
  const gateDir = path.join(opHome, 'gate');
  const bridgeDir = path.join(root, 'bridge');
  const hooksCopyDir = path.join(bridgeDir, 'hooks');
  const libCopyDir = path.join(bridgeDir, 'lib');
  const artDir = path.join(root, 'artifacts');
  const txPath = path.join(root, 'transcript.jsonl');
  for (const d of [dataDir, gateDir, hooksCopyDir, libCopyDir, artDir]) {
    fs.mkdirSync(d, { recursive: true });
  }
  fs.writeFileSync(path.join(opHome, 'mode'), 'enforce\n');
  fs.copyFileSync(path.join(HOOKS_DIR, 'operator-gate.py'), path.join(gateDir, 'operator-gate.py'));
  fs.copyFileSync(path.join(HOOKS_DIR, 'lite_core.py'), path.join(gateDir, 'lite_core.py'));
  fs.copyFileSync(path.join(HOOKS_DIR, 'session_automaton.json'), path.join(gateDir, 'session_automaton.json'));

  for (const f of ['memory.cjs', 'reflex.cjs', 'guard.cjs', 'stop-memory.cjs']) {
    fs.copyFileSync(path.join(HOOKS_DIR, f), path.join(hooksCopyDir, f));
  }
  fs.copyFileSync(path.join(LIB_DIR, 'memory-triggers.cjs'), path.join(libCopyDir, 'memory-triggers.cjs'));

  const triggersJson = path.join(bridgeDir, 'triggers.json');
  const { defaults } = require(path.join(LIB_DIR, 'memory-triggers.cjs'));
  fs.writeFileSync(triggersJson, JSON.stringify(defaults, null, 2));

  const memCfgPath = path.join(bridgeDir, 'memory-config.json');
  fs.writeFileSync(memCfgPath, JSON.stringify({
    version: '7.0.0',
    binary: VESTIGE_MCP_BIN,
    binaryArgs: ['--data-dir', dataDir],
    triggers: triggersJson,
    deadlineMs: 15000,
  }, null, 2));

  const bridgeCfgPath = path.join(bridgeDir, 'bridge.json');
  fs.writeFileSync(bridgeCfgPath, JSON.stringify({
    python: findPython(),
    gate: path.join(gateDir, 'operator-gate.py'),
    vestigeBin: VESTIGE_BIN || VESTIGE_MCP_BIN,
    dataDir,
    memoryConfig: memCfgPath,
    hooksDir: hooksCopyDir,
    gateTimeoutMs: 15000,
    memoryDeadlineMs: 15000,
    reflexDeadlineMs: 12000,
    capsuleTimeoutMs: 10000,
    debug: false,
  }, null, 2));

  const callBridge = (ev, payload) => {
    const r = spawnSync(process.execPath, [SHIPPED_BRIDGE_JS, '--event', ev, '--config', bridgeCfgPath], {
      cwd: bridgeDir,
      env: {
        ...process.env,
        VESTIGE_DATA_DIR: dataDir,
        OPERATOR_HOME: opHome,
        VESTIGE_HOOK_DEADLINE_MS: '15000',
      },
      input: JSON.stringify(payload),
      encoding: 'utf8',
      timeout: 30000,
      windowsHide: true,
    });
    assert.equal(r.status, 0, `${ev} exited ${r.status}: ${r.stderr}`);
    return JSON.parse((r.stdout || '{}').trim() || '{}');
  };

  const mcpCall = (toolName, toolArgs) => {
    const reqs = [
      JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'e2e', version: '7.0.0' } } }),
      JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
      JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: toolName, arguments: toolArgs } }),
    ].join('\n') + '\n';
    const r = spawnSync(VESTIGE_MCP_BIN, ['--data-dir', dataDir], {
      input: reqs,
      encoding: 'utf8',
      timeout: 15000,
      windowsHide: true,
    });
    assert.equal(r.status, 0, `${toolName} mcp exited ${r.status}: ${r.stderr}`);
    const lines = r.stdout.trim().split(/\r?\n/).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
    const ans = lines.find(m => m.id === 2);
    assert.ok(ans && ans.result && !ans.result.isError, `${toolName} RPC failed: ${r.stdout}`);
    return ans.result.structuredContent || JSON.parse(ans.result.content[0].text);
  };

  return {
    root,
    dataDir,
    opHome,
    bridgeDir,
    artDir,
    txPath,
    callBridge,
    mcpCall,
    cleanup: () => fs.rmSync(root, { recursive: true, force: true }),
  };
}

test('1. Shipped Antigravity bridge, Call Coach (CC-001..CC-013), Reflex & 6-event agy-hooks.json are complete and save-gate-free', () => {
  const reflexSrc = fs.readFileSync(path.join(HOOKS_DIR, 'reflex.cjs'), 'utf8');
  const memorySrc = fs.readFileSync(path.join(HOOKS_DIR, 'memory.cjs'), 'utf8');
  const bridgeSrc = fs.readFileSync(SHIPPED_BRIDGE_JS, 'utf8');

  assert.ok(reflexSrc.includes('coachPreToolUse'), 'reflex.cjs must include Call Coach');
  assert.ok(reflexSrc.includes('VALID_TOOL_KEYS'), 'reflex.cjs must protect canonical 6-tool keys from learned alias corruption');
  assert.ok(memorySrc.includes('tryDirectAttach'), 'memory.cjs must include tryDirectAttach');
  assert.ok(memorySrc.includes("lines.on('error'"), 'memory.cjs must guard Node v26 ECONNRESET stream errors');
  assert.ok(bridgeSrc.includes('intentionsStartContext'), 'agy-bridge.cjs must query active intentions');
  assert.ok(bridgeSrc.includes('capsuleRun'), 'agy-bridge.cjs must run vestige capsule');
  assert.ok(bridgeSrc.includes('postCommitReceiptRun'), 'agy-bridge.cjs must run vestige hook post-commit');
  assert.ok(bridgeSrc.includes('pendingOperatorStop'), 'agy-bridge.cjs must enforce precedent() after Operator stops');
  assert.equal(bridgeSrc.includes('saveGateRun'), false, 'saveGateRun bloat must stay removed from agy-bridge.cjs');

  const rawHooks = fs.readFileSync(SHIPPED_HOOKS_JSON);
  assert.equal(hasUtf8Bom(rawHooks), false, 'agy-hooks.json must be BOM-free');
  const parsedHooks = JSON.parse(rawHooks.toString('utf8'));
  for (const ev of ['SessionStart', 'PreToolUse', 'PostToolUse', 'PreInvocation', 'PostInvocation', 'Stop']) {
    assert.ok(Array.isArray(parsedHooks.vestige[ev]) && parsedHooks.vestige[ev].length > 0, `agy-hooks.json missing ${ev}`);
  }
});

test('2. Live Antigravity installation (when present) is BOM-free and carries all 6 events', { skip: !HAS_AGY_ENV || !fs.existsSync(AGY_HOOKS_JSON) }, () => {
  const raw = fs.readFileSync(AGY_HOOKS_JSON);
  assert.equal(hasUtf8Bom(raw), false, 'hooks.json must NOT have a UTF-8 BOM');
  const parsed = JSON.parse(raw.toString('utf8'));
  for (const ev of ['SessionStart', 'PreToolUse', 'PostToolUse', 'PreInvocation', 'PostInvocation', 'Stop']) {
    assert.ok(Array.isArray(parsed.vestige[ev]) && parsed.vestige[ev].length > 0, `Installed hooks.json missing ${ev}`);
  }
  if (fs.existsSync(AGY_BRIDGE_JS)) {
    assert.equal(
      fs.readFileSync(AGY_BRIDGE_JS, 'utf8'),
      fs.readFileSync(SHIPPED_BRIDGE_JS, 'utf8'),
      'Installed agy-bridge.cjs must match shipped packages/vestige-init/hooks/agy-bridge.cjs'
    );
  }
});

test('3. Universal @vestige/init setup plans, applies, inspects & uninstalls Antigravity hooks for any user', async () => {
  const setup = require(SETUP_LIB);
  const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-setup-test-'));
  fs.mkdirSync(path.join(tmpHome, '.gemini', 'config'), { recursive: true });
  const fakeBin = path.join(tmpHome, process.platform === 'win32' ? 'vestige-mcp.cmd' : 'vestige-mcp');
  const fakeCli = path.join(tmpHome, process.platform === 'win32' ? 'vestige.cmd' : 'vestige');
  if (process.platform === 'win32') {
    fs.writeFileSync(fakeBin, '@echo off\r\necho vestige-mcp 7.0.0\r\n');
    fs.writeFileSync(fakeCli, '@echo off\r\necho vestige 7.0.0\r\n');
  } else {
    fs.writeFileSync(fakeBin, '#!/bin/sh\necho "vestige-mcp 7.0.0"\n', { mode: 0o755 });
    fs.writeFileSync(fakeCli, '#!/bin/sh\necho "vestige 7.0.0"\n', { mode: 0o755 });
  }
  try {
    const p = setup.plan({ home: tmpHome, bin: fakeBin, platform: process.platform, clients: ['antigravity'] });
    assert.ok(!p.unsupported);
    assert.ok(p.changes.some(c => String(c.file || c.path || '').includes('vestige')), 'plan() must produce antigravity changes');
    const applied = setup.apply(p);
    assert.ok(applied.written > 0);

    const agyDir = path.join(tmpHome, '.gemini', 'config', 'plugins', 'vestige');
    assert.ok(fs.existsSync(path.join(agyDir, 'plugin.json')));
    assert.ok(fs.existsSync(path.join(agyDir, 'hooks.json')));
    assert.ok(fs.existsSync(path.join(agyDir, 'bridge', 'agy-bridge.cjs')));
    assert.ok(fs.existsSync(path.join(agyDir, 'bridge', 'hooks', 'reflex.cjs')));
    assert.ok(fs.existsSync(path.join(agyDir, 'bridge', 'hooks', 'memory.cjs')));

    const bCfg = JSON.parse(fs.readFileSync(path.join(agyDir, 'bridge', 'bridge.json'), 'utf8'));
    assert.equal(bCfg.saveGate, undefined, 'saveGate must not be configured in bridge.json');
    const mCfg = JSON.parse(fs.readFileSync(path.join(agyDir, 'bridge', 'memory-config.json'), 'utf8'));
    assert.equal(mCfg.version, '7.0.0');

    const insp = await setup.inspect(tmpHome, 5000);
    assert.ok(insp.clients.some(c => c.id === 'antigravity'));
    const agyChecks = insp.checks.filter(c => c.name.startsWith('Antigravity plugin:'));
    assert.equal(agyChecks.length, 6);
    assert.ok(agyChecks.every(c => c.ok), JSON.stringify(agyChecks));

    const again = setup.plan({ home: tmpHome, bin: fakeBin, platform: process.platform, clients: ['antigravity'] });
    assert.equal(again.changes.length, 0, 'Repeat antigravity install must be idempotent');

    setup.uninstall(tmpHome);
    assert.equal(fs.existsSync(path.join(agyDir, 'hooks.json')), false, 'uninstall removes antigravity hooks.json');
  } finally {
    fs.rmSync(tmpHome, { recursive: true, force: true });
  }
});

test('4. PreToolUse Call Coach across ALL 6 MCP tools (CC-001..CC-013 + VALID_TOOL_KEYS protection)', () => {
  const reflex = require(path.join(HOOKS_DIR, 'reflex.cjs'));

  // CC-006: recall(tag) -> handle
  const cRecall = reflex.coachPreToolUse('mcp__vestige__recall', { tag: 'pinned' });
  assert.equal(cRecall.deny, false);
  assert.equal(cRecall.updatedInput.handle, 'pinned');
  assert.equal(cRecall.updatedInput.tag, undefined);

  // CC-008: causal_walk(handle) -> from_handle
  const cWalk = reflex.coachPreToolUse('mcp__vestige__causal_walk', { handle: 'mem-12345678' });
  assert.equal(cWalk.updatedInput.from_handle, 'mem-12345678');
  assert.equal(cWalk.updatedInput.scan_limit, 500);

  // CC-007: remember(text) -> content
  const cRem = reflex.coachPreToolUse('mcp__vestige__remember', { text: 'important decision' });
  assert.equal(cRem.updatedInput.content, 'important decision');

  // CC-013: curate(action='update') -> 'edit'
  const cCurEdit = reflex.coachPreToolUse('mcp__vestige__curate', { action: 'update', id: 'mem-1', content: 'updated' });
  assert.equal(cCurEdit.updatedInput.action, 'edit');

  // CC-010 & CC-011 & CC-002 hard denies
  assert.equal(reflex.coachPreToolUse('mcp__vestige__curate', { action: 'delete', id: 'mem-1' }).deny, true);
  assert.equal(reflex.coachPreToolUse('mcp__vestige__remember', { action: 'save', content: '   ' }).deny, true);
  assert.equal(reflex.coachPreToolUse('mcp__vestige__curate', { action: 'weave', note: 'x'.repeat(501) }).deny, true);

  // CC-009 + VALID_TOOL_KEYS guard: never rename precedent.limit -> action even if bad learned rule exists
  const cPrec = reflex.coachPreToolUse(
    'mcp__vestige__precedent',
    { action: 'stops', limit: 5 },
    [{ id: 'CC-L999', leaf: 'precedent', badKey: 'limit', goodKey: 'action', derived_from: 'old' }]
  );
  assert.equal(cPrec.updatedInput.query, 'stops');
  assert.equal(cPrec.updatedInput.limit, 5);
});

test('5. UNDENIABLE E2E: Zero-Prompt Recall (SessionStart + PreInvocation) + Capsule + Intentions on live v7 binary', { skip: !VESTIGE_MCP_BIN, timeout: 45000 }, () => {
  const env = createIsolatedBridgeEnv();
  try {
    // Seed pinned rule, factual topic memory, and an active intention
    const pin = env.mcpCall('remember', {
      action: 'save',
      content: 'Standing rule: always verify WordNet hypernym expansion before tagging.',
      node_type: 'fact',
      tags: ['pinned'],
    });
    assert.ok(pin.nodeId);

    const topic = env.mcpCall('remember', {
      action: 'save',
      content: 'WordNet hypernym chains link cello and violin up to string instrument for zero-overlap recall.',
      node_type: 'fact',
      tags: ['wordnet'],
    });
    assert.ok(topic.nodeId);

    env.mcpCall('session', { action: 'set_intention', intention: 'Complete E2E verification for all 6 MCP hooks' });

    const sid = `e2e-recall-${Date.now()}`;
    const sOut = env.callBridge('SessionStart', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
    });
    const sMsg = sOut.injectSteps?.[0]?.systemMessage?.systemMessage || '';
    assert.ok(sMsg.includes('Active Intentions & Plans:'), `SessionStart missing intentions:\n${sMsg}`);
    assert.ok(sMsg.includes('capsule version='), `SessionStart missing capsule:\n${sMsg}`);
    assert.equal(sMsg.includes('alert (session)'), false, `Unexpected memory alert:\n${sMsg}`);

    const t0 = performance.now();
    const pOut = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 1,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'How do WordNet hypernym chains work for cello?',
    });
    const elapsedMs = performance.now() - t0;
    const pMsg = pOut.injectSteps?.[0]?.systemMessage?.systemMessage || '';
    assert.ok(pMsg.includes(topic.nodeId), `PreInvocation failed to recall ${topic.nodeId} in ${elapsedMs.toFixed(0)}ms:\n${pMsg}`);
    assert.ok(pMsg.includes('WordNet hypernym chains link cello'), `PreInvocation missing recalled memory content:\n${pMsg}`);
    assert.equal(pMsg.includes('alert (prompt)'), false, `Unexpected prompt memory alert:\n${pMsg}`);
  } finally {
    env.cleanup();
  }
});

test('6. UNDENIABLE E2E: Operator STOP forces mandatory precedent(query="stops") before next mutation', { skip: !VESTIGE_MCP_BIN, timeout: 45000 }, () => {
  const env = createIsolatedBridgeEnv();
  try {
    const sid = `e2e-op-${Date.now()}`;

    // Operator stops git push --force with OP-004
    const stopOut = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 1,
      toolCall: { id: 'op-1', name: 'run_command', args: { CommandLine: 'git push --force origin main' } },
    });
    assert.equal(stopOut.decision, 'deny');
    assert.match(stopOut.reason, /OP-004/);
    assert.match(stopOut.reason, /precedent\(query="stops"\)/);

    // Trying any mutation before inspecting precedent() MUST be denied
    const blockedEdit = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 2,
      toolCall: { id: 'op-2', name: 'edit_file', args: { TargetFile: 'crates/vestige-core/src/lib.rs' } },
    });
    assert.equal(blockedEdit.decision, 'deny');
    assert.match(blockedEdit.reason, /Precedent Required/);

    // Running precedent(action="stops") gets rewritten by CC-009 -> query="stops" and clears block on PostToolUse
    const prePrec = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 3,
      toolCall: { id: 'op-3', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'precedent', Arguments: { action: 'stops' } } },
    });
    assert.equal(prePrec.decision, 'allow');
    assert.equal(prePrec.overwrite?.Arguments?.query, 'stops');

    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 3,
      toolCall: { id: 'op-3', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'precedent', Arguments: { query: 'stops' } } },
    });

    const afterPrec = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 4,
      toolCall: { id: 'op-4', name: 'run_command', args: { CommandLine: 'git status' } },
    });
    assert.equal(afterPrec.decision, 'allow', JSON.stringify(afterPrec));
  } finally {
    env.cleanup();
  }
});

test('7. UNDENIABLE E2E: Mid-Conversation Anti-Spiral forces causal_walk/recall (Stage 1) then search_web/invoke_subagent (Stage 2)', { skip: !VESTIGE_MCP_BIN, timeout: 60000 }, () => {
  const env = createIsolatedBridgeEnv();
  try {
    const sid = `e2e-spiral-${Date.now()}`;
    env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 1,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Fix cargo test failure.',
    });

    // Two consecutive failing runs trigger Stage 1
    const failCall = { id: 'sp-1', name: 'run_command', args: { CommandLine: 'cargo test -p vestige-mcp' } };
    env.callBridge('PostToolUse', { conversationId: sid, workspacePaths: [REPO_ROOT], cwd: REPO_ROOT, stepIdx: 1, toolCall: failCall, error: 'Exit code 101' });
    env.callBridge('PostToolUse', { conversationId: sid, workspacePaths: [REPO_ROOT], cwd: REPO_ROOT, stepIdx: 2, toolCall: failCall, error: 'Exit code 101' });

    // Edit and repeat failing command are BOTH denied by Stage 1
    const editStage1 = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 3,
      toolCall: { id: 'sp-2', name: 'edit_file', args: { TargetFile: 'crates/vestige-mcp/src/tools/recall.rs' } },
    });
    assert.equal(editStage1.decision, 'deny');
    assert.match(editStage1.reason, /Anti-Spiral \[Stage 1\]/);

    // Calling causal_walk is allowed and clears Stage 1 block
    const walkCall = { id: 'sp-3', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'causal_walk', Arguments: { handle: 'crates/vestige-mcp/src/tools/recall.rs' } } };
    const preWalk = env.callBridge('PreToolUse', { conversationId: sid, workspacePaths: [REPO_ROOT], cwd: REPO_ROOT, stepIdx: 4, toolCall: walkCall });
    assert.equal(preWalk.decision, 'allow');
    assert.equal(preWalk.overwrite?.Arguments?.from_handle, 'crates/vestige-mcp/src/tools/recall.rs');
    env.callBridge('PostToolUse', { conversationId: sid, workspacePaths: [REPO_ROOT], cwd: REPO_ROOT, stepIdx: 4, toolCall: walkCall });

    // Edit unlocked after causal_walk
    const editAfterWalk = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 5,
      toolCall: { id: 'sp-4', name: 'edit_file', args: { TargetFile: 'crates/vestige-mcp/src/tools/recall.rs' } },
    });
    assert.equal(editAfterWalk.decision, 'allow', JSON.stringify(editAfterWalk));

    // Failing again after investigation escalates to Stage 2
    env.callBridge('PostToolUse', { conversationId: sid, workspacePaths: [REPO_ROOT], cwd: REPO_ROOT, stepIdx: 6, toolCall: failCall, error: 'Exit code 101' });

    const stage2Notice = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 2,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Fix cargo test failure.',
    });
    assert.match(stage2Notice.injectSteps?.[0]?.systemMessage?.systemMessage || '', /Anti-Spiral \[Stage 2\]/);

    const editStage2 = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 7,
      toolCall: { id: 'sp-5', name: 'edit_file', args: { TargetFile: 'crates/vestige-mcp/src/tools/recall.rs' } },
    });
    assert.equal(editStage2.decision, 'deny');
    assert.match(editStage2.reason, /Anti-Spiral \[Stage 2\]/);

    // search_web clears Stage 2 escalation; passing run clears spiral completely
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 8,
      toolCall: { id: 'sp-6', name: 'search_web', args: { query: 'rust strata file lock' } },
    });
    env.callBridge('PostToolUse', { conversationId: sid, workspacePaths: [REPO_ROOT], cwd: REPO_ROOT, stepIdx: 9, toolCall: failCall });

    const editAfterFix = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 10,
      toolCall: { id: 'sp-7', name: 'edit_file', args: { TargetFile: 'crates/vestige-mcp/src/tools/recall.rs' } },
    });
    assert.equal(editAfterFix.decision, 'allow', JSON.stringify(editAfterFix));
  } finally {
    env.cleanup();
  }
});

test('8. UNDENIABLE E2E: Proof-Carrying Finish + Curate Stale Alert + Stop Auto-Saver', { skip: !VESTIGE_MCP_BIN, timeout: 60000 }, () => {
  const env = createIsolatedBridgeEnv();
  try {
    const sid = `e2e-finish-${Date.now()}`;
    env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 1,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: env.txPath,
      artifactDirectoryPath: env.artDir,
      lastUserInput: 'From now on never skip anchor receipts before git commit.',
    });

    // Artifact edit is ignored as scratch; repo edit registers unanchored debt
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      artifactDirectoryPath: env.artDir,
      stepIdx: 1,
      toolCall: { id: 'fin-art', name: 'write_file', args: { Path: path.join(env.artDir, 'notes.md'), Content: 'scratch' } },
    });

    const absRepoFile = path.join(REPO_ROOT, 'crates', 'vestige-core', 'src', 'lib.rs');
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      artifactDirectoryPath: env.artDir,
      stepIdx: 2,
      toolCall: { id: 'fin-ed', name: 'edit_file', args: { Path: absRepoFile, Target: 'a', Replacement: 'b' } },
    });

    // git commit BEFORE remember MUST be denied by native Rust Reflex!
    const commitDenied = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 3,
      toolCall: { id: 'fin-cm1', name: 'run_command', args: { CommandLine: 'git commit -m "unanchored"' } },
    });
    assert.equal(commitDenied.decision, 'deny');
    assert.ok(commitDenied.reason.includes('crates/vestige-core/src/lib.rs'), commitDenied.reason);

    // Stop BEFORE remember MUST block (decision: 'continue')
    const stop1 = env.callBridge('Stop', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      terminationReason: 'NO_TOOL_CALL',
      continuationActive: false,
      lastAssistantMessage: 'Done.',
    });
    assert.equal(stop1.decision, 'continue', JSON.stringify(stop1));

    // Agent calls remember(files=['crates/vestige-core/src/lib.rs']) with real mem-... nodeId + contradiction alert
    fs.writeFileSync(
      env.txPath,
      JSON.stringify({
        step_index: 4,
        type: 'GENERIC',
        content: 'Created At: now\n{"decision":"create","nodeId":"mem-000000000000cafe","contradictions":["mem-000000000000old1"],"success":true}',
      }) + '\n',
      'utf8'
    );
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: env.txPath,
      stepIdx: 4,
      toolCall: {
        id: 'fin-rem',
        name: 'call_mcp_tool',
        args: { ServerName: 'vestige_vestige', ToolName: 'remember', Arguments: { action: 'save', content: 'Anchored lib.rs change', files: ['crates/vestige-core/src/lib.rs'] } },
      },
    });

    // Next PreInvocation MUST surface Vestige Curate Alert for the contradiction!
    const curatePre = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 2,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'From now on never skip anchor receipts before git commit.',
    });
    assert.match(curatePre.injectSteps?.[0]?.systemMessage?.systemMessage || '', /Vestige Curate Alert:/);

    // git commit AFTER remember MUST be ALLOWED!
    const commitAllowed = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 5,
      toolCall: { id: 'fin-cm2', name: 'run_command', args: { CommandLine: 'git commit -m "anchored"' } },
    });
    assert.equal(commitAllowed.decision, 'allow', JSON.stringify(commitAllowed));

    // curate Call Coach via live Antigravity PreToolUse bridge (CC-010 block delete, CC-013 update -> edit)
    const curateDeny = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 6,
      toolCall: { id: 'fin-cur-del', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'curate', Arguments: { action: 'delete', id: 'mem-000000000000old1' } } },
    });
    assert.equal(curateDeny.decision, 'deny');
    assert.match(curateDeny.reason, /CC-010/);

    const curateRewrite = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 7,
      toolCall: { id: 'fin-cur-upd', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'curate', Arguments: { action: 'update', id: 'mem-000000000000old1', content: 'Updated invariant' } } },
    });
    assert.equal(curateRewrite.decision, 'allow');
    assert.equal(curateRewrite.overwrite?.Arguments?.action, 'edit');

    // recall(tag=...) -> handle=... AND retired/superseded node alert via PostToolUse -> PreInvocation
    const recallRewrite = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 8,
      toolCall: { id: 'fin-rec', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'recall', Arguments: { tag: 'pinned' } } },
    });
    assert.equal(recallRewrite.decision, 'allow');
    assert.equal(recallRewrite.overwrite?.Arguments?.handle, 'pinned');

    fs.writeFileSync(
      env.txPath,
      JSON.stringify({
        step_index: 8,
        type: 'GENERIC',
        content: JSON.stringify({ nodes: [{ id: 'mem-000000000old9', retired: true, text: 'stale rule' }] }),
      }) + '\n',
      'utf8'
    );
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: env.txPath,
      stepIdx: 8,
      toolCall: { id: 'fin-rec', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'recall', Arguments: { handle: 'pinned' } } },
    });
    const staleAlert = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 3,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Verify recall retired alert.',
    });
    assert.match(staleAlert.injectSteps?.[0]?.systemMessage?.systemMessage || '', /Vestige Curate Alert: recall returned retired/);

    // Live Stop auto-save persists noteworthy findings to Strata when remember() wasn't called manually
    const autoSid = `e2e-autosave-${Date.now()}`;
    const autoToken = `AutoSaveV7Token${Date.now()}`;
    const autoTx = path.join(env.root, 'autosave-tx.jsonl');
    fs.writeFileSync(autoTx, [
      JSON.stringify({ type: 'USER_INPUT', content: 'Verify v7 hook persistence' }),
      JSON.stringify({
        type: 'PLANNER_RESPONSE',
        content: `Decision: verified ${autoToken} persistence in crates/vestige-mcp/src/tools/reflex.rs; 95/95 passed.`,
        tool_calls: [{ name: 'run_command', args: { CommandLine: 'cargo test -p vestige-mcp' } }],
      }),
    ].join('\n') + '\n');
    env.callBridge('PreInvocation', {
      conversationId: autoSid,
      invocationNum: 1,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: autoTx,
      lastUserInput: 'Verify v7 hook persistence',
    });
    env.callBridge('Stop', {
      conversationId: autoSid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: autoTx,
      terminationReason: 'NO_TOOL_CALL',
      continuationActive: false,
      lastUserInput: 'Verify v7 hook persistence',
      lastAssistantMessage: `Decision: verified ${autoToken} persistence in crates/vestige-mcp/src/tools/reflex.rs; 95/95 passed.`,
    });
    const recalledAuto = env.mcpCall('recall', { handle: 'auto-hook' });
    const autoText = JSON.stringify(recalledAuto);
    assert.ok(autoText.includes(autoToken), `Stop auto-save did not persist ${autoToken} to Strata: ${autoText}`);

    // Auto-save unit check on decision/metric findings vs casual chat
    const bridge = require(SHIPPED_BRIDGE_JS);
    assert.equal(bridge.extractAutoSaveCandidate({ userText: 'thanks', assistantText: 'You are welcome!', sawToolActivity: false }), null);
    const cand = bridge.extractAutoSaveCandidate({
      userText: 'run bench',
      assistantText: 'Decision: keep Q16.16 GeoRAG in crates/vestige-core/src/retrieval/mod.rs; BEAM 100K All@5=78.59% verified.',
      alreadyRemembered: false,
      sawToolActivity: true,
    });
    assert.ok(cand && cand.node_type === 'decision' && cand.files.includes('crates/vestige-core/src/retrieval/mod.rs'));
  } finally {
    env.cleanup();
  }
});

test('9. memory.cjs survives abrupt EOF / ECONNRESET without unhandled crash (Node v26 safety)', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-econnreset-'));
  try {
    const crashBin = path.join(tmp, 'dropper.cjs');
    fs.writeFileSync(crashBin, 'process.exit(1);\n');
    const triggers = path.join(tmp, 'triggers.json');
    fs.writeFileSync(triggers, JSON.stringify({ events: { prompt: { enabled: true } } }));
    const cfg = path.join(tmp, 'cfg.json');
    fs.writeFileSync(cfg, JSON.stringify({ binary: process.execPath, binaryArgs: [crashBin], triggers, deadlineMs: 1500 }));
    const r = spawnSync(process.execPath, [path.join(HOOKS_DIR, 'memory.cjs'), '--config', cfg, '--client', 'claude', '--event', 'prompt'], {
      input: JSON.stringify({ session_id: 'drop-1', prompt: 'hello' }),
      encoding: 'utf8',
      timeout: 4000,
      windowsHide: true,
    });
    assert.equal(r.status, 0, r.stderr);
    assert.equal(r.stderr, '');
    assert.deepEqual(JSON.parse(r.stdout), {});
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('10. Silent Watcher: benign Read fast path, sensitive Read fails closed, mid-turn PreInvocation dedup, zero read spawns in PostToolUse', { skip: !VESTIGE_MCP_BIN, timeout: 45000 }, () => {
  const hooksJson = JSON.parse(fs.readFileSync(SHIPPED_HOOKS_JSON, 'utf8'));
  const postMatcher = hooksJson.vestige.PostToolUse[0].matcher;
  const preMatcher = hooksJson.vestige.PreToolUse[0].matcher;
  assert.equal(/\bread_file\b/.test(postMatcher), false, 'PostToolUse matcher must exclude read_file');
  assert.equal(/\bview_file\b/.test(postMatcher), false, 'PostToolUse matcher must exclude view_file');
  assert.equal(/\bgrep_search\b/.test(preMatcher), false, 'PreToolUse matcher must exclude grep_search');

  const env = createIsolatedBridgeEnv();
  try {
    const bridgeCfgPath = path.join(env.bridgeDir, 'bridge.json');
    const probeCfgPath = path.join(env.bridgeDir, 'probe-bridge.json');
    const probeCfg = JSON.parse(fs.readFileSync(bridgeCfgPath, 'utf8'));
    probeCfg.gate = path.join(env.root, 'nonexistent-operator-gate.py');
    fs.writeFileSync(probeCfgPath, JSON.stringify(probeCfg, null, 2));

    const runWithCfg = (cfgPath, ev, payload) => {
      const r = spawnSync(process.execPath, [SHIPPED_BRIDGE_JS, '--event', ev, '--config', cfgPath], {
        cwd: env.bridgeDir,
        env: { ...process.env, VESTIGE_DATA_DIR: env.dataDir, OPERATOR_HOME: env.opHome },
        input: JSON.stringify(payload),
        encoding: 'utf8',
        timeout: 15000,
        windowsHide: true,
      });
      assert.equal(r.status, 0, `${ev} exited ${r.status}: ${r.stderr}`);
      return JSON.parse((r.stdout || '{}').trim() || '{}');
    };

    const sid = `e2e-silent-${Date.now()}`;
    const t0 = performance.now();
    const benign = runWithCfg(probeCfgPath, 'PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 1,
      toolCall: { id: 'read-src', name: 'read_file', args: { Path: path.join(REPO_ROOT, 'crates', 'vestige-mcp', 'src', 'main.rs') } },
    });
    const benignMs = performance.now() - t0;
    assert.deepEqual(benign, { decision: 'allow' });
    assert.ok(benignMs < 1500, `Benign Read fast path took unexpectedly long (${benignMs.toFixed(1)}ms)`);

    const sensitive = runWithCfg(probeCfgPath, 'PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 2,
      toolCall: { id: 'read-env', name: 'read_file', args: { Path: path.join(REPO_ROOT, 'config', '.env') } },
    });
    assert.equal(sensitive.decision, 'deny');
    assert.match(sensitive.reason, /Operator gate missing/i);

    // Mid-turn PreInvocation dedup: identical repeated alert injects once and suppresses exact duplicate
    env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 1,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Verify silent watcher dedup',
    });
    const opStop1 = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 3,
      toolCall: { id: 'op-d1', name: 'run_command', args: { CommandLine: 'git push --force origin main' } },
    });
    assert.equal(opStop1.decision, 'deny');
    const mid1 = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 2,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Verify silent watcher dedup',
    });
    assert.match(mid1.injectSteps?.[0]?.systemMessage?.systemMessage || '', /Operator OP-004/);

    const opStop2 = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 4,
      toolCall: { id: 'op-d2', name: 'run_command', args: { CommandLine: 'git push --force origin main' } },
    });
    assert.equal(opStop2.decision, 'deny');
    // Consume the second unique Operator stop receipt first, then test identical repeated retired-node recall alerts
    const mid2 = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 3,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Verify silent watcher dedup',
    });
    assert.match(mid2.injectSteps?.[0]?.systemMessage?.systemMessage || '', /Operator OP-004/);

    fs.writeFileSync(
      env.txPath,
      JSON.stringify({
        step_index: 5,
        type: 'GENERIC',
        content: JSON.stringify({ nodes: [{ id: 'mem-000000000old9', retired: true, text: 'stale rule' }] }),
      }) + '\n',
      'utf8'
    );
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: env.txPath,
      stepIdx: 5,
      toolCall: { id: 'dedup-r1', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'recall', Arguments: { handle: 'pinned' } } },
    });
    const dedup1 = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 4,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Verify silent watcher dedup',
    });
    assert.match(dedup1.injectSteps?.[0]?.systemMessage?.systemMessage || '', /Vestige Curate Alert: recall returned retired/);

    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: env.txPath,
      stepIdx: 6,
      toolCall: { id: 'dedup-r2', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'recall', Arguments: { handle: 'pinned' } } },
    });
    const dedup2 = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 5,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Verify silent watcher dedup',
    });
    assert.deepEqual(dedup2, {});
  } finally {
    env.cleanup();
  }
});

test('11. UNDENIABLE E2E: 6-Tool Cognitive Autopilot (Silent Call Coach + Multi-File Intention Guard + Secret Scan Commit Gate + Async SessionStart)', { skip: !VESTIGE_MCP_BIN, timeout: 45000 }, () => {
  const env = createIsolatedBridgeEnv();
  try {
    const sid = `e2e-autopilot-${Date.now()}`;

    // 1) Async SessionStart completes cleanly with capsule even on empty store
    const tBoot = performance.now();
    const sOut = env.callBridge('SessionStart', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
    });
    const bootMs = performance.now() - tBoot;
    assert.ok((sOut.injectSteps?.[0]?.systemMessage?.systemMessage || '').includes('capsule version='), `Expected capsule in ${bootMs.toFixed(0)}ms`);

    // Turn 1
    env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 1,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Build 3 modules cleanly',
    });

    // 2) Silent Call Coach: recall(tag='pinned') rewrites overwrite silently with ZERO queue banner
    const rCoach = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 1,
      toolCall: { id: 'auto-cc', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'recall', Arguments: { tag: 'pinned' } } },
    });
    assert.equal(rCoach.decision, 'allow');
    assert.equal(rCoach.overwrite?.Arguments?.handle, 'pinned');
    const afterCoach = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 2,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Build 3 modules cleanly',
    });
    assert.equal((afterCoach.injectSteps?.[0]?.systemMessage?.systemMessage || '').includes('[CC-006]'), false);

    // 3) Multi-file unplanned drift guard: 3 source edits without active intention -> single nudge
    for (const [idx, rel] of ['crates/vestige-mcp/src/a.rs', 'crates/vestige-mcp/src/b.rs', 'crates/vestige-mcp/src/c.rs'].entries()) {
      env.callBridge('PostToolUse', {
        conversationId: sid,
        workspacePaths: [REPO_ROOT],
        cwd: REPO_ROOT,
        artifactDirectoryPath: env.artDir,
        stepIdx: 2 + idx,
        toolCall: { id: `mf-${idx}`, name: 'edit_file', args: { Path: path.join(REPO_ROOT, rel), Target: 'a', Replacement: 'b' } },
      });
    }
    const intNudge = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 3,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Build 3 modules cleanly',
    });
    assert.match(intNudge.injectSteps?.[0]?.systemMessage?.systemMessage || '', /Vestige Autopilot \[session\]: 3 source files edited/);

    // 4th edit must NOT re-nag
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      artifactDirectoryPath: env.artDir,
      stepIdx: 5,
      toolCall: { id: 'mf-4', name: 'edit_file', args: { Path: path.join(REPO_ROOT, 'crates/vestige-mcp/src/d.rs'), Target: 'a', Replacement: 'b' } },
    });
    const noRepeatNudge = env.callBridge('PreInvocation', {
      conversationId: sid,
      invocationNum: 4,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      lastUserInput: 'Build 3 modules cleanly',
    });
    assert.equal((noRepeatNudge.injectSteps?.[0]?.systemMessage?.systemMessage || '').includes('Vestige Autopilot [session]'), false);

    // 4) Sensitive config edit requires precedent(query='scan_secrets') before git commit even after remember()
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      artifactDirectoryPath: env.artDir,
      stepIdx: 6,
      toolCall: { id: 'sens-ed', name: 'edit_file', args: { Path: path.join(REPO_ROOT, 'config', '.env.local'), Target: 'A=1', Replacement: 'A=2' } },
    });
    fs.writeFileSync(
      env.txPath,
      JSON.stringify({
        step_index: 7,
        type: 'GENERIC',
        content: '{"decision":"create","nodeId":"mem-000000000000beef","success":true}',
      }) + '\n',
      'utf8'
    );
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      transcriptPath: env.txPath,
      stepIdx: 7,
      toolCall: { id: 'sens-rem', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'remember', Arguments: { action: 'save', content: 'Anchored edits', files: ['crates/vestige-mcp/src/a.rs', 'crates/vestige-mcp/src/b.rs', 'crates/vestige-mcp/src/c.rs', 'crates/vestige-mcp/src/d.rs', 'config/.env.local'] } } },
    });

    const commitBlocked = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 8,
      toolCall: { id: 'sens-cm1', name: 'run_command', args: { CommandLine: 'git commit -m "chore: config"' } },
    });
    assert.equal(commitBlocked.decision, 'deny');
    assert.match(commitBlocked.reason, /scan_secrets/);

    // Clear via precedent(query='scan_secrets') -> commit now allowed!
    env.callBridge('PostToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 9,
      toolCall: { id: 'sens-scan', name: 'call_mcp_tool', args: { ServerName: 'vestige_vestige', ToolName: 'precedent', Arguments: { query: 'scan_secrets' } } },
    });
    const commitCleared = env.callBridge('PreToolUse', {
      conversationId: sid,
      workspacePaths: [REPO_ROOT],
      cwd: REPO_ROOT,
      stepIdx: 10,
      toolCall: { id: 'sens-cm2', name: 'run_command', args: { CommandLine: 'git commit -m "chore: config"' } },
    });
    assert.equal(commitCleared.decision, 'allow', JSON.stringify(commitCleared));
  } finally {
    env.cleanup();
  }
});
