'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const setup = require('../lib/vestige-memory-setup');
const memory = require('../hooks/memory.cjs');
const reflex = require('../hooks/reflex.cjs');
const { defaults } = require('../lib/memory-triggers.cjs');
const SIX = ['session', 'remember', 'recall', 'causal_walk', 'curate', 'precedent'];

function fixture(t, reflexResult = { context: 'Native Reflex context', failure_id: 'mem-0000000000000001' }) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-six-hook-'));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  const callsFile = path.join(home, 'calls.jsonl');
  const server = path.join(home, 'server.cjs');
  fs.writeFileSync(server, `#!${process.execPath}
    const fs = require('node:fs');
    if (process.argv.includes('--version')) { console.log('vestige-mcp 6.2.0'); process.exit(0); }
    require('node:readline').createInterface({ input: process.stdin }).on('line', line => {
      const msg = JSON.parse(line); if (!msg.id) return;
      let result = { protocolVersion: '2025-06-18', capabilities: {}, serverInfo: { name: 'fixture', version: '6.2.0' } };
      if (msg.method === 'tools/call') {
        const { name, arguments: args } = msg.params;
        fs.appendFileSync(${JSON.stringify(callsFile)}, JSON.stringify(msg.params) + '\\n');
        let answer;
        if (!${JSON.stringify(SIX)}.includes(name)) throw Error('legacy outgoing tool ' + name);
        if (name === 'session' && args.action === 'start' && args.memory_hook) {
          answer = { memories: [{ id: 'mem-0000000000000001', content: 'Maya has a peanut allergy', entities: ['person:Maya'] }],
            entities: ['person:Maya'], memory_count: 1, backfill_autofire: true, backfill_receipts: [{ receipt_id: 'receipt-1', memory_id: 'mem-0000000000000001', reason: 'new health fact' }] };
        } else if (name === 'remember' && args.action === 'backfill' && args.params.trigger_id === 'mem-0000000000000001') {
          answer = { change_budget_reached: false, receipts: [] };
        } else if (name === 'remember' && args.entities && args.intent_id) {
          answer = { success: true, nodeId: 'mem-0000000000000001' };
        } else if (name === 'session' && args.action === 'reflex' && args.params.action === 'hook') {
          answer = ${JSON.stringify(reflexResult)};
        } else throw Error('unexpected six-tool envelope: ' + JSON.stringify(msg.params));
        result = { structuredContent: answer };
      }
      process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result }) + '\\n');
    });
  `, { mode: 0o755 });
  const triggers = path.join(home, 'triggers.json'); fs.writeFileSync(triggers, JSON.stringify(defaults));
  const config = path.join(home, 'config.json'); fs.writeFileSync(config, JSON.stringify({ binary: server, triggers }));
  const calls = () => fs.existsSync(callsFile) ? fs.readFileSync(callsFile, 'utf8').trim().split('\n').map(JSON.parse) : [];
  return { home, server, config, calls };
}

test('all memory lifecycle events use unified calls with unchanged admission and exact-trigger replay', async t => {
  const f = fixture(t);
  for (const event of ['session', 'prompt', 'turn', 'compact', 'end', 'tool']) {
    const result = await memory.run({ client: 'claude', event, configFile: f.config,
      input: { session_id: 'six-lifecycle', prompt: event === 'prompt' ? 'Maya has a peanut allergy' : '', source: 'compact' }, deadlineMs: 2000 });
    if (['session', 'prompt'].includes(event)) {
      assert.match(result.hookSpecificOutput.additionalContext, /using remember/);
      assert.doesNotMatch(result.hookSpecificOutput.additionalContext, /smart_ingest|session_start|search tools/);
    }
  }
  const calls = f.calls();
  assert(calls.every(c => SIX.includes(c.name)));
  assert.deepEqual([...new Set(calls.map(c => c.name))].sort(), ['remember', 'session']);
  const saves = calls.filter(c => c.name === 'remember' && !c.arguments.action);
  assert.equal(saves.length, 1, 'queued facts are admitted once');
  assert.deepEqual(saves[0].arguments.entities, ['person:Maya']);
  assert.deepEqual(saves[0].arguments.attributes, ['allergy:peanut']);
  assert.match(saves[0].arguments.intent_id, /^hook:/);
  const replays = calls.filter(c => c.arguments.action === 'backfill');
  assert.deepEqual(replays, [{ name: 'remember', arguments: { action: 'backfill', params: { trigger_id: 'mem-0000000000000001' } } }]);
  const start = calls.find(c => c.arguments.memory_hook?.event === 'session');
  assert.deepEqual([start.arguments.action, start.arguments.memory_hook.session, start.arguments.memory_hook.host, start.arguments.memory_hook.source], ['start', 'six-lifecycle', 'claude', 'compact']);
  const receipts = fs.readFileSync(path.join(f.home, 'receipts.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(receipts.length, 6);
  assert.deepEqual(receipts.find(r => r.event === 'turn').saved, ['mem-0000000000000001']);
  assert.equal(fs.existsSync(path.join(f.home, 'errors.log')), false);
});

test('Reflex preserves host fields and returned context inside session action envelopes', async t => {
  const f = fixture(t);
  for (const [client, event] of [['claude', 'PostToolUseFailure'], ['codex', 'PostToolUse'], ['gemini', 'AfterTool'], ['cursor', 'postToolUse']]) {
    const input = { hook_event_name: event, session_id: 'reflex-session', agent_id: 'agent-7', tool_use_id: 'call-9', tool_name: 'Bash', cwd: '/repo',
      tool_input: { command: 'cargo test', file_path: '/repo/main.rs' }, tool_response: { exit_code: 1, stdout: 'test failed', stderr: 'failure' } };
    const output = await reflex.run({ client, configFile: f.config, input, deadlineMs: 2000 });
    const context = output.additional_context || output.hookSpecificOutput?.additionalContext;
    assert.match(context, /Native Reflex context/);
    assert.match(context, /start_points=.*logged_write.*mem-0000000000000001/);
    assert.doesNotMatch(context, /from_handle|mem-failure|automatically triggered/);
    assert.deepEqual(f.calls().at(-1), { name: 'session', arguments: { action: 'reflex', params: reflex.argumentsFor(client, event, input) } });
  }
});

test('Reflex never fabricates a failure memory when native returns no handle', async t => {
  const f = fixture(t, { context: 'Native failure context without a memory' });
  const output = await reflex.run({ client: 'claude', configFile: f.config,
    input: { hook_event_name: 'PostToolUseFailure', session_id: 'failure', error: 'boom' }, deadlineMs: 2000 });
  assert.equal(output.hookSpecificOutput.additionalContext, 'Native failure context without a memory');
});

test('doctor checks the packet through session(action=start)', async t => {
  const f = fixture(t);
  const base = path.join(f.home, '.vestige', 'memory'); fs.mkdirSync(base, { recursive: true });
  fs.writeFileSync(path.join(base, 'manifest.json'), JSON.stringify({ binary: f.server, clients: [], files: [] }));
  const report = await setup.inspect(f.home, 2000);
  assert(report.checks.find(c => c.name === 'Strata responds with memory-hook schema').ok);
  assert(report.checks.find(c => c.name === 'Backfill autofire enabled in responding process').ok);
  assert.deepEqual(f.calls(), [{ name: 'session', arguments: { action: 'start', memory_hook: { limit: 3 } } }]);
});

test('setup upgrades legacy Reflex registrations once and preserves unrelated hooks and rollback', t => {
  const f = fixture(t); fs.mkdirSync(path.join(f.home, '.claude'));
  const file = path.join(f.home, '.claude', 'settings.json');
  const legacy = setup.reflexHooks('claude', 'unused').find(([event]) => event === 'PreToolUse')[1];
  legacy.hooks[0] = { ...legacy.hooks[0], tool: 'reflex', input: legacy.hooks[0].input.params };
  const other = { type: 'mcp_tool', server: 'other', tool: 'reflex', input: { preserve: true } };
  const mixed = { hooks: [{ type: 'command', command: 'user-hook' }, other, { type: 'mcp_tool', server: 'vestige', tool: 'reflex', input: { event: 'Stop', session: 'custom-session' } }] };
  const original = { hooks: { PreToolUse: [legacy], Stop: [mixed] }, theme: 'keep' };
  fs.writeFileSync(file, JSON.stringify(original));
  const opts = { home: f.home, platform: process.platform, env: { PATH: '/usr/bin:/bin', VESTIGE_MCP_PATH: f.server }, clients: ['claude'] };
  setup.apply(setup.plan(opts));
  const installed = JSON.parse(fs.readFileSync(file));
  assert.equal(installed.hooks.PreToolUse.length, 1, 'the migrated default is not appended twice');
  assert.equal(installed.hooks.PreToolUse[0].hooks[0].tool, 'session');
  assert.deepEqual(installed.hooks.Stop[0].hooks[1], other);
  assert.deepEqual(installed.hooks.Stop[0].hooks[2].input, { action: 'reflex', params: { event: 'Stop', session: 'custom-session' } });
  assert.equal(setup.plan(opts).changes.length, 0);
  installed.userAdded = true; fs.writeFileSync(file, JSON.stringify(installed));
  setup.uninstall(f.home);
  assert.deepEqual(JSON.parse(fs.readFileSync(file)), { ...original, userAdded: true });
});

test('Operator STOP relay writes through remember and preserves its native node id', () => {
  const script = path.join(__dirname, '../hooks/operator-stop-memory.py');
  const result = spawnSync('python3', ['-c', `import runpy, sys
module = runpy.run_path(sys.argv[1])
class MCP:
    def tool(self, name, args):
        assert name == 'remember', name
        assert args == {'items': [{'content': 'stop fixture'}]}, args
        return {'results': [{'nodeId': 'mem-0000000000000001'}]}
assert module['ingest'](MCP(), {'content': 'stop fixture'}) == 'mem-0000000000000001'
`, script], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
});
