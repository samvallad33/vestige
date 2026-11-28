'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const setup = require('../lib/vestige-memory-setup');
const { rpc } = require('../hooks/memory.cjs');
const BIN = process.env.VESTIGE_TEST_MCP_BIN;
const SIX = ['causal_walk', 'curate', 'precedent', 'recall', 'remember', 'session'];
// Observe the actual wire from every installed command shim and doctor process,
// including their independent MCP children. No native tool responses are mocked.
function traceBinary(home, t) {
  const binary = path.join(home, 'traced-vestige-mcp');
  const trace = path.join(home, 'outgoing-tools.jsonl');
  // A tee adds no language-runtime startup inside the hook's 90/190ms budget.
  // Decode after execution so the observer does not change the measured work.
  const quote = value => "'" + value.replace(/'/g, "'\"'\"'") + "'";
  fs.writeFileSync(binary, `#!/bin/sh\nif [ "$1" = "--version" ]; then exec ${quote(path.resolve(BIN))} "$@"; fi\ntee -a ${quote(trace)} | ${quote(path.resolve(BIN))} "$@"\n`, { mode: 0o755 });
  const calls = () => fs.existsSync(trace) ? fs.readFileSync(trace, 'utf8').trim().split('\n').map(JSON.parse).filter(row=>row.method==='tools/call').map(row=>row.params) : [];
  t.after(() => {
    const rows = calls();
    assert(rows.length > 0, 'native MCP calls were observed');
    assert(rows.every(row => SIX.includes(row.name)), JSON.stringify(rows.filter(row => !SIX.includes(row.name))));
    t.diagnostic(JSON.stringify({ outgoing_tools: [...new Set(rows.map(row=>row.name))].sort(), calls: rows.length, trace }));
  });
  return { binary, calls };
}
test('native hooks: first-chat Maya, real Backfill receipt, all lifecycle timing, doctor', { skip: !BIN, timeout: 30000 }, async t => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-memory-native-'));
  const wire = traceBinary(home, t);
  for (const name of ['.claude','.codex','.gemini','.cursor']) fs.mkdirSync(path.join(home, name));
  const store = path.join(home, 'store');
  const env = { ...process.env, HOME: home, USERPROFILE: home, VESTIGE_DATA_DIR: store,
    OPERATOR_HOME: path.join(home,'.operator'),
    VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0', VESTIGE_BACKFILL_AUTOFIRE: '1', VESTIGE_MCP_PATH: wire.binary, VESTIGE_HOOK_DIAGNOSTICS: '1' };
  const p = setup.plan({ home, platform: process.platform, env }); setup.apply(p);
  const config = path.join(p.base, 'config.json');
  const c = JSON.parse(fs.readFileSync(config)); c.binaryArgs = ['--no-http','--data-dir',store]; fs.writeFileSync(config, JSON.stringify(c));
  const mcp = rpc(wire.binary, ['--no-http','--data-dir',store], 20000, env); t.after(() => mcp.close()); await mcp.ready();
  await mcp.tool('session',{action:'start',memory_hook:{limit:3}});
  t.diagnostic(`owner endpoint available: ${fs.existsSync(path.join(store,'.serve.endpoint'))}; diagnostics: ${mcp.diagnostics()}`);
  const times = [];
  const hook = (event, prompt = '', session = 'maya-birthday') => {
    const before = performance.now();
    const r = spawnSync(process.execPath, [path.join(p.base,'hooks','memory.cjs'), '--config',config,'--client','claude','--event',event],
      { input: JSON.stringify({ session_id:session,prompt }), env, encoding:'utf8', timeout:2000 });
    const elapsed = performance.now()-before; times.push({ event,ms:Number(elapsed.toFixed(2)) });
    assert.equal(r.status,0,r.stderr); assert.equal(r.stderr,'');
    if (elapsed >= 300) t.diagnostic(JSON.stringify({home,event,stdout:r.stdout,errors:fs.existsSync(path.join(p.base,'errors.log')) ? fs.readFileSync(path.join(p.base,'errors.log'),'utf8') : null}));
    assert(elapsed < 300, `${event}: ${elapsed}ms`);
    return JSON.parse(r.stdout);
  };
  hook('session');
  hook('prompt','Maya felt off after dinner at Bangkok Garden'); hook('turn');
  let packet = await mcp.tool('session',{action:'start',memory_hook:{prompt:'Maya',limit:8}});
  const reaction = packet.memories.find(n=>n.content.includes('Bangkok Garden'));
  assert(reaction,JSON.stringify(packet)); await mcp.tool('curate',{action:'memory',params:{action:'demote',id:reaction.id}});
  await mcp.tool('remember',{content:'Liam felt off after dinner at Bangkok Garden',entities:['person:Liam','place:Bangkok Garden']});
  await mcp.tool('remember',{content:'The bicycle tire was repaired',entities:['service:bicycle']});
  // Another connected agent may have saved the words without explicit metadata.
  await mcp.tool('remember',{content:'Maya has a peanut allergy'});
  hook('prompt','Maya has a peanut allergy'); hook('turn');
  const existing = hook('prompt','where should we take Maya for her birthday dinner?');
  const exposed = (existing.hookSpecificOutput?.additionalContext || '').split('\n').filter(line=>line.startsWith('{')).map(JSON.parse);
  assert(!exposed.some(row=>row.id===reaction.id), 'the same session already received the reaction on the allergy prompt');
  const answer = hook('prompt','where should we take Maya for her birthday dinner?', 'maya-birthday-new-agent');
  const context = answer.hookSpecificOutput?.additionalContext;
  assert(context,`birthday recall returned no context; ${fs.existsSync(path.join(p.base,'errors.log'))?fs.readFileSync(path.join(p.base,'errors.log'),'utf8'):'no hook errors'}`);
  assert(context.includes('Bangkok Garden'),context); assert(!context.includes('Liam'),context);
  assert(context.includes('backfill_receipt'),context);
  assert.equal(JSON.parse(context.split('\n').find(line=>line.startsWith('{'))).id,reaction.id,'raised reaction ranks first, including when the trigger is an echo');
  const receipts = await mcp.tool('remember',{action:'backfill',params:{action:'receipts'}});
  const receipt = receipts.receipts.find(r=>r.memory_id===reaction.id);
  assert.equal(receipt.shared_entity,'person:Maya'); assert.match(receipt.reason,/new health fact about Maya/);
  const recorded = fs.readFileSync(path.join(p.base,'receipts.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
  const last = recorded.at(-1); assert(last.trigger.matches.some(m=>m.category==='questions'&&m.phrase==='where'));
  assert(last.trigger.matches.some(m=>m.category==='entity'&&m.phrase==='person:Maya'));
  assert(last.backfill_receipts.includes(receipt.receipt_id));
  hook('compact'); hook('end');
  const doctor = spawnSync(process.execPath,[path.join(__dirname,'../bin/init.js'),'doctor','--json'],{env,encoding:'utf8',timeout:5000});
  assert.equal(doctor.status,0,doctor.stdout+doctor.stderr); const health=JSON.parse(doctor.stdout); assert(health.checks.every(c=>c.ok),doctor.stdout);
  const reflexHook = spawnSync(process.execPath, [path.join(p.base,'hooks','reflex.cjs'), '--config', config, '--client', 'claude'],
    { input: JSON.stringify({ hook_event_name: 'SessionStart', session_id: 'native-six-reflex', source: 'startup' }), env, encoding: 'utf8', timeout: 2000 });
  assert.equal(reflexHook.status, 0, reflexHook.stderr);
  assert.equal(reflexHook.stderr, '');
  assert.doesNotThrow(() => JSON.parse(reflexHook.stdout));
  assert(wire.calls().some(row => row.name === 'session' && row.arguments.action === 'reflex' && row.arguments.params.session === 'native-six-reflex'));
  const reflexStatus = await mcp.tool('session', { action: 'reflex', params: { action: 'status' } });
  for (const field of ['version', 'sessions', 'receiptChainBreaks']) assert(Object.hasOwn(reflexStatus, field), JSON.stringify(reflexStatus));
  t.diagnostic(JSON.stringify({hook_wall_times:times,memory_count:health.memory_count,receipt:receipt.receipt_id,doctor:health.checks}));
});

test('native Rust gate: guarded STOP writes SHA-256 day-file receipt verified by precedent', { skip: !BIN, timeout:30000 }, async t => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(),'vestige-lite-native-'));
  const wire = traceBinary(home, t);
  const vestigeBin = path.join(path.dirname(BIN), process.platform === 'win32' ? 'vestige.exe' : 'vestige');
  if (!fs.existsSync(vestigeBin)) return;
  const operatorHome = path.join(home,'.operator');
  fs.mkdirSync(operatorHome,{recursive:true}); fs.mkdirSync(path.join(home,'.claude'));
  fs.writeFileSync(path.join(operatorHome,'mode'),'enforce\n');
  const store = path.join(home,'store'); const env = {...process.env,HOME:home,USERPROFILE:home,OPERATOR_HOME:operatorHome,VESTIGE_DATA_DIR:store,VESTIGE_MCP_PATH:wire.binary,VESTIGE_HTTP_ENABLED:'0',VESTIGE_DASHBOARD_ENABLED:'false',VESTIGE_AUTOPILOT_ENABLED:'0'};
  const p = setup.plan({home,platform:process.platform,env,guard:true}); setup.apply(p);
  const mcp=rpc(wire.binary,['--no-http','--data-dir',store],20000,env); t.after(()=>mcp.close()); await mcp.ready(); await mcp.tool('session',{action:'start',memory_hook:{limit:3}});
  const guardBase=path.join(operatorHome,'state','vestige-memory');
  const cfgPath = path.join(guardBase, 'config.json');
  fs.writeFileSync(cfgPath, JSON.stringify({ vestigeBin, gate: 'native' }));
  for (const file of [cfgPath, path.join(guardBase,'guard.cjs')]) {
    const write=spawnSync(process.execPath,[path.join(guardBase,'guard.cjs'),'--config',cfgPath,'--client','claude'],{input:JSON.stringify({hook_event_name:'PreToolUse',session_id:'free-loop-fixture',cwd:home,tool_name:'Write',tool_input:{file_path:file,content:'bypass the guard'}}),env,encoding:'utf8',timeout:15000});
    assert.equal(write.status,2,`Guard runtime must protect ${path.basename(file)}: ${write.stdout} ${write.stderr}`);
  }
  const payload={hook_event_name:'PreToolUse',session_id:'free-loop-fixture',cwd:home,tool_name:'Bash',tool_input:{command:'git push --force origin main'}};
  const stopped=spawnSync(process.execPath,[path.join(guardBase,'guard.cjs'),'--config',cfgPath,'--client','claude'],{input:JSON.stringify(payload),env,encoding:'utf8',timeout:15000});
  assert.equal(stopped.status,2,stopped.stdout+stopped.stderr);
  const files=fs.readdirSync(path.join(operatorHome,'receipts')).filter(f=>f.endsWith('.jsonl'));
  const rows=files.flatMap(f=>fs.readFileSync(path.join(operatorHome,'receipts',f),'utf8').trim().split('\n').map(JSON.parse));
  const receipt=rows.find(r=>r.decision==='STOP' && r.commitment==='OP-004'); assert(receipt,JSON.stringify(rows));
  const verify=spawnSync(vestigeBin,['gate','verify'],{env,encoding:'utf8',timeout:5000});
  assert.equal(verify.status,0,verify.stderr);
  assert.match(verify.stdout,/chain=OK/);
  const prec=await mcp.tool('precedent',{query:'stops',rule:'OP-004'});
  assert(prec.items && prec.items.length >= 1, JSON.stringify(prec));
  assert.equal(prec.items[0].digest, receipt.digest);
  assert.equal(prec.chain.breaks_total, 0);
  const count=(await mcp.tool('session',{action:'start',memory_hook:{limit:3}})).memory_count;
  assert.equal((await mcp.tool('session',{action:'start',memory_hook:{limit:3}})).memory_count,count);
  t.diagnostic(JSON.stringify({rule:receipt.commitment,gate_receipt:receipt.digest.slice(0,16)}));
});
