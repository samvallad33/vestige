'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const setup = require('../lib/vestige-memory-setup');
const { defaults, trigger, contains } = require('../lib/memory-triggers.cjs');
const memory = require('../hooks/memory.cjs');
function fixture() {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-memory-test-'));
  const bin = path.join(home, 'vestige-mcp');
  fs.writeFileSync(bin, '#!/bin/sh\nprintf "vestige-mcp 5.0.1\\n"\n', { mode: 0o755 });
  const env = { PATH: '/usr/bin:/bin', VESTIGE_MCP_PATH: bin };
  for (const name of ['.claude', '.codex', '.gemini', '.cursor']) fs.mkdirSync(path.join(home, name));
  return { home, bin, env };
}
test('fresh install previews all files; repeat changes nothing; exact uninstall restores backups', () => {
  const f = fixture(); const settings = path.join(f.home, '.claude', 'settings.json');
  const original = '{ "hooks": {"Stop":[{"hooks":[{"type":"command","command":"my-original-hook"}]}]}, "theme":"dark" }\n';
  fs.writeFileSync(settings, original);
  const first = setup.plan({ ...f, platform: 'darwin' });
  assert.equal(fs.existsSync(first.manifestFile), false);
  assert(first.changes.some(c => c.file === settings));
  const applied = setup.apply(first); assert(applied.written >= 8); assert(applied.backups.length > 0);
  assert(JSON.parse(fs.readFileSync(settings)).hooks.Stop.some(g => g.hooks[0].command === 'my-original-hook'));
  assert.equal(JSON.parse(fs.readFileSync(settings)).hooks.PreToolUse, undefined, 'memory only never adds a gate');
  const again = setup.plan({ ...f, platform: 'darwin' }); assert.equal(again.changes.length, 0);
  assert.equal(setup.apply(again).written, 0);
  const result = setup.uninstall(f.home); assert.equal(result.kept.length, 0);
  assert.equal(fs.readFileSync(settings, 'utf8'), original);
  assert.equal(fs.existsSync(first.manifestFile), false);
});
test('a client config with comments is left untouched and setup continues for the others (#515)', () => {
  const f = fixture(); const zed = path.join(f.home, '.config', 'zed', 'settings.json');
  fs.mkdirSync(path.dirname(zed), { recursive: true });
  const original = '// Zed settings\n//\n// For information on how to configure Zed, see the Zed docs\n{\n  "theme": "One Dark",\n}\n';
  fs.writeFileSync(zed, original);
  const p = setup.plan({ ...f, platform: 'linux' });
  assert.deepEqual(p.handAdd.map(h => h.id), ['zed']);
  assert.equal(p.handAdd[0].file, zed);
  assert(!p.detected.some(c => c.id === 'zed'));
  assert(!p.changes.some(c => c.file === zed), 'a commented config is never rewritten');
  assert(p.detected.some(c => c.id === 'claude'), 'the other clients still get set up');
  setup.apply(p);
  assert.equal(fs.readFileSync(zed, 'utf8'), original);
});
test('an empty client config file is treated as empty JSON, not a crash', () => {
  const f = fixture(); const hooks = path.join(f.home, '.cursor', 'hooks.json');
  fs.writeFileSync(hooks, ''); fs.writeFileSync(path.join(f.home, '.cursor', 'mcp.json'), '{}');
  const p = setup.plan({ ...f, platform: 'linux' });
  assert(p.detected.some(c => c.id === 'cursor'));
  assert(p.changes.some(c => c.file === hooks));
});
test('uninstall preserves hooks and properties added after installation', () => {
  const f = fixture(); const p = setup.plan({ ...f, platform: 'linux' }); setup.apply(p);
  const file = path.join(f.home, '.cursor', 'hooks.json'); const value = JSON.parse(fs.readFileSync(file));
  value.newUserProperty = 'keep'; value.hooks.sessionStart.push({ command: 'user-added-later' });
  fs.writeFileSync(file, JSON.stringify(value));
  setup.uninstall(f.home);
  const restored = JSON.parse(fs.readFileSync(file));
  assert.equal(restored.newUserProperty, 'keep');
  assert.deepEqual(restored.hooks.sessionStart, [{ command: 'user-added-later' }]);
});
test('preview drift and invalid config refuse before any change', () => {
  const f = fixture(); const p = setup.plan({ ...f, platform: 'linux' });
  const file = path.join(f.home, '.claude.json'); fs.writeFileSync(file, '{"user":"new"}');
  assert.throws(() => setup.apply(p), /Changed since preview/);
  assert.equal(fs.existsSync(p.manifestFile), false);
  fs.writeFileSync(file, '{invalid'); assert.throws(() => setup.plan({ ...f, platform: 'linux' }));
});
test('missing binary is a clean no-write result; platform catalog and shell paths are explicit', () => {
  const f = fixture(); const p = setup.plan({ ...f, platform: 'win32', env: { PATH: '', VESTIGE_MCP_PATH: path.join(f.home, 'missing') } });
  assert.equal(p.missing, true); assert.deepEqual(p.changes, []);
  assert(setup.catalog(f.home, 'win32', { PATH: '', APPDATA: path.join(f.home, 'AppData', 'Roaming') }).some(c => c.name === 'Codex'));
  assert.equal(setup.quote("/tmp/it's a path", 'linux'), "'/tmp/it'\"'\"'s a path'");
  assert.equal(setup.quote('C:\\Users\\A B\\node.exe', 'win32'), '"C:\\Users\\A B\\node.exe"');
  assert.throws(() => setup.quote('C:\\bad%PATH%', 'win32'));
});
test('all configured trigger phrases use case insensitive Unicode word boundaries', () => {
  for (const [category, phrases] of Object.entries(defaults)) for (const phrase of phrases) {
    const result = trigger(`Hey, ${phrase.toUpperCase()}?`);
    assert(result.matches.some(m => m.category === category && m.phrase === phrase), phrase);
  }
  assert.equal(contains('somewhereabouts', 'where'), false);
  assert.equal(contains('thematic', 'them'), false);
  assert.equal(trigger('hello there').mode, 'light'); assert.equal(trigger('hello there').limit, 3);
  assert.equal(trigger('Maya', {}, ['person:Maya']).matches[0].category, 'entity');
  assert.equal(trigger('custom exact', { mine: ['custom exact'] }).mode, 'triggered');
});
test('explicit statement capture records the named person and place without inferring ingredients', () => {
  assert.deepEqual(memory.explicitFact('Maya felt off after dinner at Bangkok Garden').entities, ['person:Maya','place:Bangkok Garden']);
  assert.deepEqual(memory.explicitFact('Maya has a peanut allergy').attributes, ['allergy:peanut']);
  assert.equal(memory.explicitFact('where should we take Maya?'), null);
  assert.equal(memory.explicitFact('run this command'), null);
  assert.equal(memory.explicitFact('Maya is allergic?'), null);
  assert.deepEqual(memory.output('cursor', 'prompt', 'context'), {}, 'do not invent unsupported Cursor prompt output');
});
test('memory hook fails open silently when MCP is unavailable, within 300ms', async () => {
  const f = fixture(); const p = setup.plan({ ...f, platform: 'linux' }); setup.apply(p);
  const configFile = path.join(p.base, 'config.json'); const c = JSON.parse(fs.readFileSync(configFile));
  c.binary = path.join(f.home, 'does-not-exist'); fs.writeFileSync(configFile, JSON.stringify(c));
  const started = performance.now(); const result = await memory.run({ event: 'prompt', client: 'claude', configFile, input: { session_id: 'offline', prompt: 'Where is Maya?' } });
  const elapsed = performance.now() - started;
  assert.deepEqual(result, {}); assert(elapsed < 300, `${elapsed}ms`);
  assert(fs.readFileSync(path.join(p.base, 'errors.log'), 'utf8').includes('prompt:'));
});
test('guard is opt-in, keeps an installed gate, and adapter fails closed', { skip: process.platform === 'win32' }, () => {
  const f = fixture(); const gate = path.join(f.home, '.operator', 'gate', 'operator-gate.py');
  fs.mkdirSync(path.dirname(gate), { recursive: true }); fs.copyFileSync(path.join(__dirname, '..', 'hooks', 'operator-lite.py'), gate);
  const p = setup.plan({ ...f, platform: 'linux', guard: true });
  assert(!p.changes.some(c => c.file === gate), 'installed gate is preserved'); setup.apply(p);
  const file = path.join(f.home, '.claude', 'settings.json'); const v = JSON.parse(fs.readFileSync(file));
  assert(v.hooks.PreToolUse.length > 0);
  const configFile = path.join(f.home, '.operator', 'state', 'vestige-memory', 'config.json'); const c = JSON.parse(fs.readFileSync(configFile)); c.gate = path.join(f.home, 'missing-guard'); fs.writeFileSync(configFile, JSON.stringify(c));
  const result = spawnSync(process.execPath, [path.join(f.home, '.operator', 'state', 'vestige-memory', 'guard.cjs'), '--config', configFile, '--client', 'codex'], { input: '{}', encoding: 'utf8' });
  assert.equal(result.status, 2); assert.match(result.stderr, /Operator stopped/);
});
test('bundled free Lite and stop reflex are exact copies of public sources', () => {
  assert.equal(fs.readFileSync(path.join(__dirname, '../hooks/operator-lite.py'), 'utf8'), fs.readFileSync(path.join(__dirname, '../../../operator-lite/operator-gate.py'), 'utf8'));
  assert.equal(fs.readFileSync(path.join(__dirname, '../hooks/operator-stop-memory.py'), 'utf8'), fs.readFileSync(path.join(__dirname, '../../../hooks/operator-stop-memory.py'), 'utf8'));
});

test('uninstall retains runtime files referenced by a user-edited guard hook', { skip: process.platform === 'win32' }, () => {
  const f=fixture(); const gate=path.join(f.home,'.operator','gate','operator-gate.py'); fs.mkdirSync(path.dirname(gate),{recursive:true});
  fs.copyFileSync(path.join(__dirname,'../hooks/operator-lite.py'),gate);
  const p=setup.plan({...f,platform:'linux',guard:true}); setup.apply(p);
  const file=path.join(f.home,'.claude','settings.json'); const value=JSON.parse(fs.readFileSync(file));
  value.hooks.PreToolUse[0].hooks[0].command += ' --user-customization'; fs.writeFileSync(file,JSON.stringify(value));
  const result=setup.uninstall(f.home);
  assert(result.kept.includes(path.join(f.home,'.operator','state','vestige-memory','guard.cjs')));
  assert(fs.existsSync(path.join(f.home,'.operator','state','vestige-memory','guard.cjs')));
  assert(fs.existsSync(path.join(p.base,'config.json')));
  assert(JSON.parse(fs.readFileSync(file)).hooks.PreToolUse[0].hooks[0].command.includes('--user-customization'));
});

test('an existing guard registration stays single so it consumes a permit only once', { skip: process.platform === 'win32' }, () => {
  const f=fixture(); const gate=path.join(f.home,'.operator','gate','operator-gate.py'); fs.mkdirSync(path.dirname(gate),{recursive:true});
  fs.copyFileSync(path.join(__dirname,'../hooks/operator-lite.py'),gate);
  const file=path.join(f.home,'.claude','settings.json');
  const existing={matcher:'*',hooks:[{type:'command',command:`python3 '${gate}' hook --source claude`}]};
  fs.writeFileSync(file,JSON.stringify({hooks:{PreToolUse:[existing]}}));
  const p=setup.plan({...f,platform:'linux',guard:true}); setup.apply(p);
  assert.deepEqual(JSON.parse(fs.readFileSync(file)).hooks.PreToolUse,[existing]);
});

test('guard recognition requires native gate argv and a catch-all matcher', () => {
  const gate='/tmp/owner/.operator/gate/operator-gate.py';
  const group=(command,matcher='*')=>[{matcher,hooks:[{type:'command',command}]}];
  assert(setup.nativeGuardRegistered(group(`python3 '${gate}' hook --source claude`),gate,'claude'));
  assert(setup.nativeGuardRegistered(group(`[ -f '${gate}' ] && python3 '${gate}' hook --source claude`),gate,'claude'));
  assert(!setup.nativeGuardRegistered(group(`echo python3 '${gate}' hook --source claude`),gate,'claude'));
  assert(!setup.nativeGuardRegistered(group(`python3 '${gate}' hook --source claude`,'Write'),gate,'claude'));
  assert(!setup.nativeGuardRegistered(group('my-guard-hook'),gate,'claude'));
  const windowsGate='C:/Users/A B/.operator/gate/operator-gate.py';
  const code=`import os,runpy,sys;g='${windowsGate}';sys.argv=[g,'hook','--source','claude'];os.path.isfile(g) and runpy.run_path(g,run_name='__main__')`;
  assert(setup.nativeGuardRegistered(group(`python -I -c "${code}"`),windowsGate,'claude'));
});

test('installed adapter consumes one fixture permit once for the exact stopped action', {skip:process.platform==='win32'}, () => {
  const f=fixture(); const operator=path.join(f.home,'.operator'); const gate=path.join(operator,'gate','operator-gate.py');
  fs.mkdirSync(path.dirname(gate),{recursive:true}); fs.copyFileSync(path.join(__dirname,'../hooks/operator-lite.py'),gate);
  fs.writeFileSync(path.join(operator,'mode'),'enforce\n');
  const p=setup.plan({...f,platform:'linux',guard:true}); setup.apply(p);
  const env={...process.env,HOME:f.home,USERPROFILE:f.home,OPERATOR_HOME:operator};
  const guardBase=path.join(operator,'state','vestige-memory');
  const payload={cwd:f.home,tool_name:'Bash',tool_input:{command:'git push --force origin main'},session_id:'permit-fixture'};
  const judge=input=>spawnSync(process.execPath,[path.join(guardBase,'guard.cjs'),'--config',path.join(guardBase,'config.json'),'--client','claude'],{input:JSON.stringify(input),env,encoding:'utf8'});
  assert.equal(judge(payload).status,2);
  const rows=fs.readdirSync(path.join(operator,'receipts')).filter(n=>n.endsWith('.jsonl')).flatMap(n=>fs.readFileSync(path.join(operator,'receipts',n),'utf8').trim().split('\n').map(JSON.parse));
  const stopped=rows.find(r=>r.decision==='STOP'); assert(stopped);
  const config=JSON.parse(fs.readFileSync(path.join(guardBase,'config.json')));
  const ownerOnly=spawnSync(config.python,['-I',gate,'approve',stopped.action_digest],{input:'ALLOW ONCE\n',env,encoding:'utf8'});
  assert.equal(ownerOnly.status,3,'piped text never grants an owner permit');
  // Test fixture admission only. This does not exercise or impersonate a human TTY.
  const permits=path.join(operator,'permits'); fs.mkdirSync(permits,{recursive:true});
  const permit=path.join(permits,stopped.action_digest+'.json');
  fs.writeFileSync(permit,JSON.stringify({action_digest:stopped.action_digest,expires:Date.now()/1000+60,single_use:true,by:'test-fixture'}),{mode:0o600});
  const other=path.join(f.home,'other-work'); fs.mkdirSync(other);
  assert.equal(judge({...payload,cwd:other}).status,2,'permit cannot authorize another working directory');
  assert(fs.existsSync(permit),'different action did not consume the original permit');
  assert.equal(judge(payload).status,0,'exact fixture action allowed once');
  assert(fs.existsSync(permit+'.used'));
  assert.equal(judge(payload).status,2,'replay stops after one use');
});

test('registered stop-memory relay fails open under 300ms with a stalled server', {skip:process.platform==='win32'}, () => {
  const f=fixture(); const gate=path.join(f.home,'.operator','gate','operator-gate.py');
  fs.mkdirSync(path.dirname(gate),{recursive:true}); fs.copyFileSync(path.join(__dirname,'../hooks/operator-lite.py'),gate);
  const p=setup.plan({...f,platform:'linux',guard:true}); setup.apply(p);
  const file=path.join(f.home,'.claude','settings.json');
  const hooks=JSON.parse(fs.readFileSync(file)).hooks;
  const command=hooks.PostToolUse.find(g=>g.hooks[0].command.includes('stop-memory.cjs')).hooks[0].command;
  const env={...process.env,HOME:f.home,OPERATOR_HOME:path.join(f.home,'.operator')};
  const stop=spawnSync(hooks.PreToolUse[0].hooks[0].command,{shell:true,input:JSON.stringify({tool_name:'Bash',cwd:f.home,tool_input:{command:'git push --force origin main'}}),env,encoding:'utf8'});
  assert.equal(stop.status,2,stop.stderr);
  fs.writeFileSync(f.bin,'#!/bin/sh\nexec sleep 5\n',{mode:0o755});
  const started=performance.now();
  const result=spawnSync(command,{shell:true,input:'{}',env,encoding:'utf8',timeout:1000});
  assert.equal(result.status,0,result.stderr); assert.equal(result.stderr,'');
  assert(performance.now()-started<300);
});

for (const responds of [true, false]) {
  test(`memory CLI bounds child shutdown after ${responds ? 'success' : 'deadline'}`, () => {
    const f = fixture(); const server = path.join(f.home, 'slow-shutdown.cjs');
    // A child that keeps its pipes open after SIGTERM used to hold the hook
    // beyond its deadline even after a valid response had been written.
    fs.writeFileSync(server, `
      const readline = require('node:readline');
      process.on('SIGTERM', () => setTimeout(() => process.exit(0), 500));
      setInterval(() => {}, 1000);
      readline.createInterface({ input: process.stdin }).on('line', line => {
        const msg = JSON.parse(line); if (!msg.id || !${responds}) return;
        const result = msg.method === 'initialize'
          ? { protocolVersion: '2025-06-18', capabilities: {}, serverInfo: { name: 'fixture', version: '5.0.1' } }
          : { structuredContent: { memories: [], entities: [], backfill_receipts: [] } };
        process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result }) + '\\n');
      });
    `);
    const triggers = path.join(f.home, 'triggers.json'); fs.writeFileSync(triggers, JSON.stringify(defaults));
    const config = path.join(f.home, 'config.json');
    fs.writeFileSync(config, JSON.stringify({ binary: process.execPath, binaryArgs: [server], triggers }));
    const start = performance.now();
    const result = spawnSync(process.execPath, [path.join(__dirname, '../hooks/memory.cjs'), '--config', config, '--client', 'claude', '--event', 'session'],
      { input: JSON.stringify({ session_id: 'slow-shutdown' }), encoding: 'utf8', timeout: 2000 });
    const elapsed = performance.now() - start;
    assert.equal(result.status, 0, result.stderr); assert.equal(result.stderr, '');
    assert(elapsed < 300, `complete hook process took ${elapsed}ms`);
    const value = JSON.parse(result.stdout);
    if (responds) assert(value.hookSpecificOutput?.additionalContext, result.stdout);
    else {
      assert.deepEqual(value, {});
      assert(fs.readFileSync(path.join(f.home, 'errors.log'), 'utf8').includes('deadline exceeded'));
    }
  });
}
test('the memory hook names its session so the server never injects a memory twice', () => {
  const f = fixture(); const server = path.join(f.home, 'recording.cjs'); const seen = path.join(f.home, 'calls.jsonl');
  fs.writeFileSync(server, `
    const fs = require('node:fs'); const readline = require('node:readline');
    readline.createInterface({ input: process.stdin }).on('line', line => {
      const msg = JSON.parse(line); if (!msg.id) return;
      if (msg.method === 'tools/call') fs.appendFileSync(${JSON.stringify(seen)}, JSON.stringify(msg.params) + '\\n');
      const result = msg.method === 'initialize'
        ? { protocolVersion: '2025-06-18', capabilities: {}, serverInfo: { name: 'fixture', version: '6.0.0' } }
        : { structuredContent: { memories: [], entities: [], backfill_receipts: [] } };
      process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result }) + '\\n');
    });
  `);
  const triggers = path.join(f.home, 'triggers.json'); fs.writeFileSync(triggers, JSON.stringify(defaults));
  const config = path.join(f.home, 'config.json');
  fs.writeFileSync(config, JSON.stringify({ binary: process.execPath, binaryArgs: [server], triggers }));
  const result = spawnSync(process.execPath, [path.join(__dirname, '../hooks/memory.cjs'), '--config', config, '--client', 'claude', '--event', 'session'],
    { input: JSON.stringify({ session_id: 's-518', source: 'compact' }), encoding: 'utf8', timeout: 4000 });
  assert.equal(result.status, 0, result.stderr);
  const call = fs.readFileSync(seen, 'utf8').trim().split('\n').map(JSON.parse).find(p => p.name === 'session_start');
  assert.deepEqual([call.arguments.memory_hook.session, call.arguments.memory_hook.host, call.arguments.memory_hook.event, call.arguments.memory_hook.source],
    ['s-518', 'claude', 'session', 'compact']);
});
function reflexFixture(version) {
  const f = fixture();
  fs.writeFileSync(f.bin, `#!/bin/sh\nprintf "vestige-mcp ${version}\\n"\n`, { mode: 0o755 });
  return f;
}
test('reflex hooks register only for a 6.0.0 or newer server', () => {
  assert.equal(setup.reflexReady(fixture().bin, { PATH: '/usr/bin:/bin' }), false);
  assert.equal(setup.reflexReady(reflexFixture('6.0.0').bin, { PATH: '/usr/bin:/bin' }), true);
  assert.equal(setup.reflexReady(reflexFixture('6.1.2').bin, { PATH: '/usr/bin:/bin' }), true);
  assert.equal(setup.reflexReady(reflexFixture('5.9.9').bin, { PATH: '/usr/bin:/bin' }), false);
  const old = fixture(); setup.apply(setup.plan({ ...old, platform: 'darwin' }));
  const oldSettings = JSON.parse(fs.readFileSync(path.join(old.home, '.claude', 'settings.json')));
  assert.equal(JSON.stringify(oldSettings).includes('"reflex"'), false, 'a 5.x server gets no reflex hooks');
});
test('claude gets mcp_tool reflex hooks plus a SessionStart shim; nothing is a gate', () => {
  const f = reflexFixture('6.0.0'); setup.apply(setup.plan({ ...f, platform: 'darwin' }));
  const hooks = JSON.parse(fs.readFileSync(path.join(f.home, '.claude', 'settings.json'))).hooks;
  for (const event of ['PreToolUse', 'PostToolUse', 'PostToolUseFailure', 'UserPromptSubmit', 'SubagentStart', 'SubagentStop', 'PreCompact', 'PostCompact', 'Stop']) {
    const handlers = hooks[event].flatMap(g => g.hooks).filter(h => h.type === 'mcp_tool');
    assert.equal(handlers.length, 1, event);
    assert.equal(handlers[0].server, 'vestige'); assert.equal(handlers[0].tool, 'reflex');
    assert.equal(handlers[0].input.session, '${session_id}'); assert.equal(handlers[0].input.call, '${tool_use_id}');
    assert.equal(handlers[0].input.agent, '${agent_id}', 'each read lands in the context of the agent that made it');
    assert.equal(handlers[0].input.error, '${error}', 'Failure Pairing reads the error the tool reported');
  }
  assert(hooks.PreToolUse.every(g => g.hooks.every(h => h.type === 'mcp_tool')), 'memory plus reflex adds no command gate');
  const shim = hooks.SessionStart.flatMap(g => g.hooks).find(h => h.command && h.command.includes('reflex.cjs'));
  assert(shim && shim.command.includes("'--client' 'claude'"));
  assert(fs.existsSync(path.join(f.home, '.vestige', 'memory', 'hooks', 'reflex.cjs')));
  const again = setup.plan({ ...f, platform: 'darwin' }); assert.equal(again.changes.length, 0, 'repeat install changes nothing');
  setup.uninstall(f.home);
  assert.equal(fs.existsSync(path.join(f.home, '.claude', 'settings.json')) && fs.readFileSync(path.join(f.home, '.claude', 'settings.json'), 'utf8').includes('reflex'), false);
});
test('codex, gemini and cursor reach reflex through the command shim', () => {
  const f = reflexFixture('6.0.0'); setup.apply(setup.plan({ ...f, platform: 'linux' }));
  const codex = JSON.parse(fs.readFileSync(path.join(f.home, '.codex', 'hooks.json'))).hooks;
  assert(codex.PreToolUse.some(g => g.matcher === '*' && g.hooks.some(h => h.command.includes('reflex.cjs'))));
  const gemini = JSON.parse(fs.readFileSync(path.join(f.home, '.gemini', 'settings.json'))).hooks;
  assert(gemini.AfterTool.some(g => g.hooks.some(h => h.command.includes("'--client' 'gemini'"))));
  const cursor = JSON.parse(fs.readFileSync(path.join(f.home, '.cursor', 'hooks.json'))).hooks;
  assert(cursor.postToolUse.some(h => h.command.includes('reflex.cjs') && h.failClosed === false));
});
test('the reflex shim maps host events and answers in each host shape', () => {
  const shim = require('../hooks/reflex.cjs');
  assert.equal(shim.canonical('gemini', 'AfterTool'), 'PostToolUse');
  assert.equal(shim.canonical('cursor', 'beforeSubmitPrompt'), 'UserPromptSubmit');
  assert.equal(shim.canonical('claude', 'SessionStart'), 'SessionStart');
  assert.equal(shim.canonical('gemini', 'PreCompress'), 'PreCompact');
  assert.equal(shim.canonical('cursor', 'preCompact'), 'PreCompact');
  assert.equal(shim.argumentsFor('claude', 'PostCompact', { session_id: 's1', compact_summary: 'kept mem-00000000000000a1', trigger: 'auto' }).summary, 'kept mem-00000000000000a1');
  const args = shim.argumentsFor('codex', 'PreToolUse', { session_id: 's1', tool_name: 'Bash', tool_use_id: 't1', tool_input: { command: 'rm -rf x' } });
  assert.deepEqual([args.event, args.session, args.call, args.command, args.format], ['PreToolUse', 's1', 't1', 'rm -rf x', 'plain']);
  assert.equal(shim.argumentsFor('codex', 'PreToolUse', { session_id: 's1', agent_id: 'ag1' }).agent, 'ag1');
  const failed = shim.argumentsFor('codex', 'PostToolUseFailure', { session_id: 's1', error: 'Exit code 1\nboom', is_interrupt: true, cwd: '/repo', tool_input: { file_path: '/repo/a.rs' } });
  assert.deepEqual([failed.error, failed.interrupt, failed.file, failed.cwd], ['Exit code 1\nboom', 'true', '/repo/a.rs', '/repo']);
  assert.equal(shim.argumentsFor('codex', 'PreToolUse', { session_id: 's1' }).agent, '');
  const ran = shim.argumentsFor('codex', 'PostToolUse', { session_id: 's1', tool_input: { command: 'cargo test', run_in_background: true }, tool_response: { stdout: ' ok\n', stderr: '', interrupted: false } });
  assert.deepEqual([ran.stdout, ran.stderr, ran.interrupted, ran.background], [' ok\n', '', '', 'true']);
  assert.deepEqual(shim.hostOutput('claude', 'SessionStart', 'hi'), { hookSpecificOutput: { hookEventName: 'SessionStart', additionalContext: 'hi' } });
  assert.deepEqual(shim.hostOutput('cursor', 'postToolUse', 'hi'), { additional_context: 'hi' });
  assert.deepEqual(shim.hostOutput('cursor', 'stop', 'hi'), {});
  assert.deepEqual(shim.hostOutput('gemini', 'AfterTool', ''), {});
});
test('the reflex shim fails open when the server is missing', async () => {
  const shim = require('../hooks/reflex.cjs');
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-reflex-shim-'));
  const config = path.join(home, 'config.json'); fs.writeFileSync(config, JSON.stringify({ binary: path.join(home, 'missing-binary') }));
  const out = await shim.run({ client: 'claude', configFile: config, input: { hook_event_name: 'PostToolUse', session_id: 's1' }, deadlineMs: 200 });
  assert.deepEqual(out, {});
});
