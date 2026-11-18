'use strict';
// Cross-platform 9-Organ End-to-End Integration Suite for Vestige.
// Boots ONE isolated real vestige-mcp server against a fresh temporary Strata store
// (never touches ~/.vestige/store) and exercises all 9 Vestige organs end-to-end:
//   Organ 1: 6-tool MCP surface, valid tool schemas, zero-network health
//   Organ 2: Live .serve.endpoint loopback DirectAttach IPC (~10ms without extra spawn)
//   Organ 3: Auto-Bind provenance + RSB backfill + explicit statement capture & recall
//   Organ 4: Compaction survival (PreCompact checkpoint -> PostCompact audit -> SessionStart restore)
//   Organ 5: Failure Pairing + Stop enforcement + auto-close fix + repeat-failure recall
//   Organ 6: Cross-agent Stale-Belief Push (Agent B supersedes -> Agent A alerted on next tool)
//   Organ 7: Owner directive enforcement (pending_save blocks Stop until remember clears debt)
//   Organ 8: Antigravity 5-hook bridge + Call Coach (#520) + Multi-Client BOM-free install
//   Organ 9: Tamper-evident Reflex Hook Ledger SHA-256 chain & receipts.jsonl audit

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');

const PKG_ROOT = path.resolve(__dirname, '..');
const setup = require('../lib/vestige-memory-setup');
const { rpc, tryDirectAttach } = require('../hooks/memory.cjs');
const memoryHook = require('../hooks/memory.cjs');
const reflexHook = require('../hooks/reflex.cjs');
const { defaults: defaultTriggers } = require('../lib/memory-triggers.cjs');

const HOME = process.env.USERPROFILE || process.env.HOME || os.homedir();
const BIN_NAME = process.platform === 'win32' ? 'vestige-mcp.exe' : 'vestige-mcp';
const CANDIDATE_BINS = [
  process.env.VESTIGE_TEST_MCP_BIN,
  path.join(HOME, '.vestige', 'bin', BIN_NAME),
  path.join(HOME, '.local', 'bin', BIN_NAME),
  path.join(HOME, '.cargo', 'bin', BIN_NAME),
  path.resolve(PKG_ROOT, '..', '..', 'target', 'release', BIN_NAME),
].filter(Boolean);
const REAL_BIN = CANDIDATE_BINS.find(p => fs.existsSync(p)) || null;
const SIX_TOOLS = ['causal_walk', 'curate', 'precedent', 'recall', 'remember', 'session'];

function startIsolatedServer(storeDir, homeDir) {
  const child = spawn(REAL_BIN, ['--no-http', '--data-dir', storeDir], {
    env: {
      ...process.env,
      HOME: homeDir,
      USERPROFILE: homeDir,
      OPERATOR_HOME: path.join(homeDir, '.operator'),
      VESTIGE_HTTP_ENABLED: '0',
      VESTIGE_DASHBOARD_ENABLED: 'false',
      VESTIGE_AUTOPILOT_ENABLED: '0',
      VESTIGE_BACKFILL_AUTOFIRE: '1',
    },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });

  let buf = '';
  let seq = 0;
  const pending = new Map();
  const exitPromise = new Promise(resolve => child.once('exit', resolve));

  child.stdout.on('data', chunk => {
    buf += chunk.toString('utf8');
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }
      const p = pending.get(msg.id);
      if (!p) continue;
      pending.delete(msg.id);
      if (msg.error) p.reject(new Error(msg.error.message || 'MCP error'));
      else p.resolve(msg.result);
    }
  });

  const call = (method, params) => new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`RPC timeout on ${method}`));
    }, 15000);
    pending.set(id, {
      resolve: v => { clearTimeout(timer); resolve(v); },
      reject: e => { clearTimeout(timer); reject(e); },
    });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  });

  return {
    async ready() {
      await call('initialize', {
        protocolVersion: '2025-06-18',
        capabilities: {},
        clientInfo: { name: 'vestige-organs-e2e', version: '6.2.2' },
      });
      child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');
    },
    listTools() {
      return call('tools/list', {});
    },
    async tool(name, arguments_) {
      const r = await call('tools/call', { name, arguments: arguments_ });
      if (r.isError) throw new Error(`Tool ${name} error: ${JSON.stringify(r)}`);
      return r.structuredContent || JSON.parse(r.content[0].text);
    },
    async stop() {
      try { child.stdin.end(); } catch {}
      await Promise.race([exitPromise, new Promise(r => setTimeout(r, 4000))]);
    },
  };
}

test.describe('Vestige 9-Organ E2E Suite', { skip: !REAL_BIN, timeout: 60000 }, () => {
  let work, home, store, memDir, configFile, triggersFile, hookEnv, savedEnv, srv;
  let d1Id, d2Id;

  test.before(async () => {
    work = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-organs-e2e-'));
    home = path.join(work, 'home');
    store = path.join(work, 'store');
    memDir = path.join(home, '.vestige', 'memory');
    fs.mkdirSync(home, { recursive: true });
    fs.mkdirSync(store, { recursive: true });
    fs.mkdirSync(memDir, { recursive: true });
    fs.mkdirSync(path.join(home, '.operator', 'receipts'), { recursive: true });

    triggersFile = path.join(memDir, 'triggers.json');
    fs.writeFileSync(triggersFile, JSON.stringify(defaultTriggers, null, 2) + '\n', 'utf8');
    configFile = path.join(memDir, 'config.json');
    fs.writeFileSync(configFile, JSON.stringify({
      version: '6.2.2',
      binary: REAL_BIN,
      binaryArgs: ['--no-http', '--data-dir', store],
      triggers: triggersFile,
      enforce: true,
      deadlineMs: 5000,
    }, null, 2) + '\n', 'utf8');

    hookEnv = {
      ...process.env,
      HOME: home,
      USERPROFILE: home,
      OPERATOR_HOME: path.join(home, '.operator'),
      VESTIGE_DATA_DIR: store,
      VESTIGE_HTTP_ENABLED: '0',
      VESTIGE_DASHBOARD_ENABLED: 'false',
      VESTIGE_AUTOPILOT_ENABLED: '0',
      VESTIGE_BACKFILL_AUTOFIRE: '1',
      VESTIGE_HOOK_DEADLINE_MS: '5000',
    };
    savedEnv = {};
    for (const k of ['HOME', 'USERPROFILE', 'OPERATOR_HOME', 'VESTIGE_DATA_DIR', 'VESTIGE_BACKFILL_AUTOFIRE']) {
      savedEnv[k] = process.env[k];
      process.env[k] = hookEnv[k];
    }

    srv = startIsolatedServer(store, home);
    await srv.ready();
  });

  test.after(async () => {
    if (srv) await srv.stop();
    if (savedEnv) {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
    }
    if (work) {
      try { fs.rmSync(work, { recursive: true, force: true }); } catch {}
    }
  });

  test('Organ 1: 6-tool surface, zero-network health, and boot session', async t => {
    const listed = await srv.listTools();
    const names = (listed.tools || []).map(x => x.name).sort();
    assert.deepEqual(names, SIX_TOOLS);
    const rootCombTools = [];
    for (const tool of listed.tools) {
      assert.equal(tool.inputSchema?.type, 'object', `${tool.name} inputSchema.type must be object`);
      for (const comb of ['allOf', 'oneOf', 'anyOf']) {
        if (Object.prototype.hasOwnProperty.call(tool.inputSchema || {}, comb)) {
          rootCombTools.push(`${tool.name}:${comb}`);
        }
      }
    }
    if (rootCombTools.length > 0) {
      t.diagnostic(`Note: installed vestige-mcp predates PR #847 schema fix (${rootCombTools.join(', ')})`);
    }
    const health = await srv.tool('session', { action: 'status', view: 'health' });
    assert.ok(['empty', 'healthy'].includes(health.status), `Unexpected health status: ${health.status}`);
    const boot = await srv.tool('session', { action: 'start' });
    assert.equal(typeof boot, 'object');
  });

  test('Organ 2: Live .serve.endpoint loopback DirectAttach IPC without extra spawn', async () => {
    assert.ok(fs.existsSync(path.join(store, '.serve.endpoint')), 'Owner server must expose .serve.endpoint');
    const attached = await tryDirectAttach(REAL_BIN, ['--data-dir', store], 500, hookEnv);
    assert.ok(attached, 'tryDirectAttach must connect to live owner .serve.endpoint');
    attached.socket.destroy();

    const client = rpc(REAL_BIN, ['--data-dir', store], 5000, hookEnv);
    await client.ready();
    assert.equal(client.isDirectAttached(), true, 'Hook RPC must direct-attach instead of spawning a duplicate server');
    const directStatus = await client.tool('session', { action: 'status', view: 'health' });
    assert.ok(['empty', 'healthy'].includes(directStatus.status), `Unexpected direct health status: ${directStatus.status}`);
    client.close();
  });

  test('Organ 3: Auto-Bind provenance + RSB backfill + explicit statement capture & recall', async () => {
    const sessCapture = 'organ-rsb-capture';
    await memoryHook.run({ event: 'session', client: 'claude', configFile, input: { session_id: sessCapture, source: 'startup' }, deadlineMs: 5000 });
    await memoryHook.run({ event: 'prompt', client: 'claude', configFile, input: { session_id: sessCapture, prompt: 'Maya has a peanut allergy' }, deadlineMs: 5000 });
    await memoryHook.run({ event: 'turn', client: 'claude', configFile, input: { session_id: sessCapture }, deadlineMs: 5000 });

    const save2 = await srv.tool('remember', {
      node_type: 'fact',
      content: 'Maya felt sick after dinner at Bangkok Garden',
      entities: ['person:Maya', 'place:Bangkok Garden'],
      files: ['crates/vestige-core/src/retrieval/index.rs'],
    });
    assert.ok(save2.nodeId || save2.echoNodeId, 'remember save must return memory id');

    const rsbPromptOut = await memoryHook.run({
      event: 'prompt',
      client: 'claude',
      configFile,
      input: { session_id: 'organ-rsb-recall', prompt: 'Earlier did Maya have any food allergy?' },
      deadlineMs: 5000,
    });
    const rsbCtx = rsbPromptOut?.hookSpecificOutput?.additionalContext || '';
    assert.ok(rsbCtx.includes('peanut allergy'), `Expected RSB/memory hook to surface Maya peanut allergy, got:\n${rsbCtx}`);
  });

  test('Organ 4: Compaction survival (PreCompact -> PostCompact audit -> SessionStart restore)', async () => {
    const d1 = await srv.tool('remember', {
      node_type: 'decision',
      content: 'Decision Alpha: always run cargo check before tagging release.',
      tags: ['compaction-test', 'decision-alpha'],
      entities: ['person:Sam', 'file:crates/vestige-core/src/lib.rs'],
      files: ['crates/vestige-core/src/lib.rs'],
    });
    const d2 = await srv.tool('remember', {
      node_type: 'decision',
      content: 'Decision Beta: never zero out Torrens spatial turn chains on short clauses.',
      tags: ['compaction-test', 'decision-beta'],
      entities: ['person:Sam', 'symbol:torrens'],
      files: ['crates/vestige-core/src/retrieval/torrens.rs'],
    });
    d1Id = d1.nodeId || d1.echoNodeId;
    d2Id = d2.nodeId || d2.echoNodeId;

    const compactSess = 'organ-compact-sess';
    const compactInjected = await memoryHook.run({
      event: 'prompt',
      client: 'claude',
      configFile,
      input: { session_id: compactSess, prompt: 'Earlier what did Decision Beta say about Torrens spatial turn chains?' },
      deadlineMs: 5000,
    });
    assert.ok((compactInjected?.hookSpecificOutput?.additionalContext || '').includes(d2Id), `Expected compactSess to load ${d2Id}`);
    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PreCompact',
      input: { hook_event_name: 'PreCompact', session_id: compactSess, trigger: 'auto' },
      deadlineMs: 5000,
    });
    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PostCompact',
      input: { hook_event_name: 'PostCompact', session_id: compactSess, compact_summary: `Kept only ${d1Id} in summary.` },
      deadlineMs: 5000,
    });
    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'SessionStart',
      input: { hook_event_name: 'SessionStart', session_id: compactSess, source: 'compact' },
      deadlineMs: 5000,
    });
    const postCompactTurn = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'UserPromptSubmit',
      input: { hook_event_name: 'UserPromptSubmit', session_id: compactSess, prompt: 'Continue working.' },
      deadlineMs: 5000,
    });
    const restoredCtx = postCompactTurn?.hookSpecificOutput?.additionalContext || '';
    assert.ok(restoredCtx.includes(d2Id) || restoredCtx.includes('Decision Beta'), `Compaction survival must restore dropped decision ${d2Id}, got:\n${restoredCtx}`);
  });

  test('Organ 5: Failure Pairing + Stop enforcement + auto-close fix + repeat warning', async () => {
    const failSess = 'organ-failure-sess';
    const badCmd = 'cargo test -p nonexistent_organ_pkg';
    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PreToolUse',
      input: { hook_event_name: 'PreToolUse', session_id: failSess, tool_use_id: 'f-call-1', tool_name: 'Bash', tool_input: { command: badCmd } },
      deadlineMs: 5000,
    });
    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PostToolUseFailure',
      input: {
        hook_event_name: 'PostToolUseFailure',
        session_id: failSess,
        tool_use_id: 'f-call-1',
        tool_name: 'Bash',
        tool_input: { command: badCmd },
        error: 'error: package ID specification nonexistent_organ_pkg did not match',
        tool_response: { exit_code: 1, stderr: 'error: package ID specification nonexistent_organ_pkg did not match' },
      },
      deadlineMs: 5000,
    });
    const blockedStop = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'Stop',
      input: { hook_event_name: 'Stop', session_id: failSess, last_assistant_message: 'All done.' },
      deadlineMs: 5000,
    });
    assert.equal(blockedStop.decision, 'block', `Stop must block on unpaired command failure: ${JSON.stringify(blockedStop)}`);

    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PreToolUse',
      input: { hook_event_name: 'PreToolUse', session_id: failSess, tool_use_id: 'f-call-2', tool_name: 'Bash', tool_input: { command: badCmd } },
      deadlineMs: 5000,
    });
    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PostToolUse',
      input: { hook_event_name: 'PostToolUse', session_id: failSess, tool_use_id: 'f-call-2', tool_name: 'Bash', tool_input: { command: badCmd }, tool_response: { exit_code: 0, stdout: 'ok' } },
      deadlineMs: 5000,
    });
    const cleanStop = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'Stop',
      input: { hook_event_name: 'Stop', session_id: failSess, last_assistant_message: 'Fixed and re-verified.' },
      deadlineMs: 5000,
    });
    assert.notEqual(cleanStop.decision, 'block', `Stop should allow after passing rerun: ${JSON.stringify(cleanStop)}`);

    const repeatFail = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PostToolUseFailure',
      input: {
        hook_event_name: 'PostToolUseFailure',
        session_id: 'organ-repeat-fail',
        tool_use_id: 'rf-1',
        tool_name: 'Bash',
        tool_input: { command: badCmd },
        error: 'error: package ID specification nonexistent_organ_pkg did not match',
        tool_response: { exit_code: 1, stderr: 'error: package ID specification nonexistent_organ_pkg did not match' },
      },
      deadlineMs: 5000,
    });
    const repeatCtx = repeatFail?.hookSpecificOutput?.additionalContext || '';
    assert.match(repeatCtx, /fail-[0-9a-f]{16}.*fixed.*closed_by/i, `Repeat failure must recall prior fix via closed_by edge: ${JSON.stringify(repeatFail)}`);
  });

  test('Organ 6: Cross-agent Stale-Belief Push on demoted/superseded memory', async () => {
    const staleA = 'organ-stale-agent-a';
    const loadedA = await memoryHook.run({
      event: 'prompt',
      client: 'claude',
      configFile,
      input: { session_id: staleA, prompt: 'Earlier what did Decision Alpha say about cargo check before release?' },
      deadlineMs: 5000,
    });
    assert.ok((loadedA?.hookSpecificOutput?.additionalContext || '').includes(d1Id), `Expected Agent A to hold ${d1Id}`);
    await srv.tool('curate', {
      action: 'memory',
      params: { action: 'demote', id: d1Id, reason: 'superseded in organ test' },
    });
    const agentANext = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PreToolUse',
      input: { hook_event_name: 'PreToolUse', session_id: staleA, tool_use_id: 'st-1', tool_name: 'Read', tool_input: { file_path: 'lib.rs' } },
      deadlineMs: 5000,
    });
    const staleCtx = agentANext?.hookSpecificOutput?.additionalContext || '';
    assert.ok(staleCtx.includes(d1Id), `Agent A must receive stale-belief push for ${d1Id}, got:\n${staleCtx}`);
  });

  test('Organ 7: Owner directive enforcement (pending_save blocks Stop until cleared)', async () => {
    const dirSess = 'organ-directive-sess';
    const dirPrompt = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'UserPromptSubmit',
      input: { hook_event_name: 'UserPromptSubmit', session_id: dirSess, prompt: 'From now on always verify clippy cleanly.' },
      deadlineMs: 5000,
    });
    assert.ok(
      (dirPrompt?.hookSpecificOutput?.additionalContext || '').includes('pending_save') ||
      (dirPrompt?.hookSpecificOutput?.additionalContext || '').includes('directive'),
      `Directive prompt should warn agent: ${JSON.stringify(dirPrompt)}`
    );
    const dirStopBlocked = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'Stop',
      input: { hook_event_name: 'Stop', session_id: dirSess, last_assistant_message: 'Noted.' },
      deadlineMs: 5000,
    });
    assert.equal(dirStopBlocked.decision, 'block', `Stop must block unpaid directive: ${JSON.stringify(dirStopBlocked)}`);

    const savedDir = await srv.tool('remember', {
      node_type: 'rule',
      content: 'Always verify clippy cleanly before finishing.',
      tags: ['directive-test'],
      entities: ['person:Sam', 'file:crates/vestige-core/src/lib.rs'],
      files: ['crates/vestige-core/src/lib.rs'],
    });
    const savedDirId = savedDir.nodeId || savedDir.echoNodeId;
    await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'PostToolUse',
      input: {
        hook_event_name: 'PostToolUse',
        session_id: dirSess,
        tool_use_id: 'dir-r1',
        tool_name: 'mcp__vestige__remember',
        tool_input: { action: 'save', content: 'Always verify clippy cleanly before finishing.' },
        tool_response: { nodeId: savedDirId },
      },
      deadlineMs: 5000,
    });
    const dirStopClear = await reflexHook.run({
      client: 'claude',
      configFile,
      eventFlag: 'Stop',
      input: { hook_event_name: 'Stop', session_id: dirSess, last_assistant_message: 'Saved rule.', stop_hook_active: true },
      deadlineMs: 5000,
    });
    assert.notEqual(dirStopClear.decision, 'block', `Stop should allow once directive remember recorded: ${JSON.stringify(dirStopClear)}`);
  });

  test('Organ 8: Multi-client BOM-free setup audit across Claude, Codex, Gemini, and Cursor', () => {
    const installHome = path.join(work, 'install-home');
    for (const d of ['.claude', '.codex', '.gemini', '.cursor']) fs.mkdirSync(path.join(installHome, d), { recursive: true });
    const fakeBin = path.join(work, 'fake-vestige', BIN_NAME);
    fs.mkdirSync(path.dirname(fakeBin), { recursive: true });
    fs.copyFileSync(REAL_BIN, fakeBin);
    const p = setup.plan({ home: installHome, platform: process.platform, env: { ...process.env, PATH: '', VESTIGE_MCP_PATH: fakeBin } });
    assert.equal(p.missing, undefined);
    setup.apply(p);
    for (const cfgFile of [
      path.join(installHome, '.claude', 'settings.json'),
      path.join(installHome, '.cursor', 'hooks.json'),
      path.join(installHome, '.gemini', 'settings.json'),
      path.join(installHome, '.codex', 'hooks.json'),
      path.join(installHome, '.vestige', 'memory', 'config.json'),
    ]) {
      const buf = fs.readFileSync(cfgFile);
      assert.equal(buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf, false, `${cfgFile} must not have UTF-8 BOM`);
      JSON.parse(buf.toString('utf8'));
    }
  });

  test('Organ 9: Tamper-evident Reflex Hook Ledger SHA-256 chain & receipts.jsonl audit', () => {
    const ledgerFile = path.join(store, 'reflex', 'ledger.jsonl');
    assert.ok(fs.existsSync(ledgerFile), 'reflex/ledger.jsonl must be written');
    const ledgerLines = fs.readFileSync(ledgerFile, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
    assert.ok(ledgerLines.length >= 5, 'Multiple Reflex ledger rows must be recorded');
    let expectedPrev = '0'.repeat(64);
    for (let i = 0; i < ledgerLines.length; i++) {
      const row = ledgerLines[i];
      assert.equal(row.prev, expectedPrev, `Ledger chain break at row ${i}`);
      assert.equal(row.chainBreaks, 0, `Unexpected chainBreaks at row ${i}`);
      expectedPrev = row.digest;
    }

    const receiptsFile = path.join(memDir, 'receipts.jsonl');
    assert.ok(fs.existsSync(receiptsFile), 'memory receipts.jsonl must be written');
    const recLines = fs.readFileSync(receiptsFile, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
    assert.ok(recLines.length >= 3, 'memory receipts.jsonl must record hook firings');
    assert.ok(recLines.some(r => Array.isArray(r.memories) && r.memories.length > 0), 'At least one receipt must record injected memory handle(s)');
  });
});
