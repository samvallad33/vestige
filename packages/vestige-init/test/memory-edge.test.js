'use strict';
// $0 edge cases for hook injection against the real binary (VESTIGE_TEST_MCP_BIN).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs'); const os = require('os'); const path = require('path');
const m = require('../hooks/memory.cjs');
const BIN = process.env.VESTIGE_TEST_MCP_BIN;
const ids = out => out.split('\n').filter(l => l.startsWith('{"id"')).map(l => JSON.parse(l).id);
async function store(t) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-edge-'));
  const dir = path.join(home, 'store');
  const env = { ...process.env, HOME: home, VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0' };
  const mcp = m.rpc(BIN, ['--no-http', '--data-dir', dir], 20000, env); t.after(() => mcp.close()); await mcp.ready();
  const add = (content, entities = []) => mcp.tool('smart_ingest', { content, entities });
  const ask = async prompt => { const p = await mcp.tool('session_start', { memory_hook: { prompt, limit: 8, trigger: {}, session: 's' + Math.random(), host: 'claude', event: 'prompt', source: '' } }); return { p, out: m.formatContext(p) }; };
  return { add, ask };
}
const opt = { skip: !BIN, timeout: 60000 };
test('edge: empty store injects no record and does not crash', opt, async t => {
  const s = await store(t); const { p, out } = await s.ask('What did I say about my grant?');
  assert.ok(Array.isArray(p.memories)); assert.equal(ids(out).length, 0);
});
test('edge: no match injects no record', opt, async t => {
  const s = await store(t); await s.add('The bicycle tire was repaired');
  assert.equal(ids((await s.ask('zyzzyva quokka xylophone')).out).length, 0);
});
test('edge: true record outranks a near-miss distractor', opt, async t => {
  const s = await store(t);
  await s.add('I went to the page about the park and the project and the things we did for the objectives of the day');
  await s.add('Grant aim page: molecular subtypes of endometrial cancer, three objectives outlined');
  const { p } = await s.ask('Remind me of the three objectives for the grant aim page on molecular subtypes and endometrial cancer');
  assert.match(p.memories[0].content, /endometrial/);
});
test('edge: overflow truncates by rank and stays under the cap', opt, async t => {
  const s = await store(t);
  for (let i = 0; i < 12; i++) await s.add(`Marathon training log ${i}: ` + 'long run notes '.repeat(150 + i));
  const { p, out } = await s.ask('marathon training log');
  assert.ok(out.length <= 7000, String(out.length));
  const shown = ids(out); assert.ok(shown.length >= 1);
  assert.deepEqual(shown, p.memories.map(r => r.id).filter(id => shown.includes(id)));
  assert.match(out, /smart_ingest/);
});
test('edge: superseded fact — newest first, old still available', opt, async t => {
  const s = await store(t);
  await s.add('Maya lives in Austin', ['person:Maya', 'place:Austin']);
  await s.add('Maya moved to Denver', ['person:Maya', 'place:Denver']);
  const { p } = await s.ask('Where does Maya live now? Austin or Denver?');
  const texts = p.memories.map(r => r.content);
  assert.ok(texts.indexOf('Maya moved to Denver') === 0, JSON.stringify(texts));
  assert.ok(texts.includes('Maya lives in Austin'), JSON.stringify(texts));
});
test('edge: case-mismatched duplicate person tags', opt, async t => {
  const s = await store(t);
  await s.add('Maya likes jazz', ['person:Maya']); await s.add('maya plays cello', ['person:maya']);
  const { p, out } = await s.ask('What does Maya like?');
  assert.ok(p.memories.some(r => /jazz/.test(r.content))); assert.equal(new Set(ids(out)).size, ids(out).length);
});
for (const [name, prompt] of [['multi-line', 'Line one about nothing.\n\nWhat instrument\ndoes Maya play?'], ['non-English/emoji', '¿Qué instrumento toca Maya? 🎻🎶'], ['very long', 'filler words '.repeat(4000) + ' What instrument does Maya play?']]) {
  test(`edge: ${name} prompt`, opt, async t => {
    const s = await store(t); await s.add('Maya plays cello', ['person:Maya']);
    const { out } = await s.ask(prompt);
    assert.ok(out.length <= 7000); assert.ok(ids(out).length >= 1, out);
  });
}
test('edge: formatContext on an empty or malformed packet never throws', () => {
  for (const p of [{}, { memories: [] }, { memories: [{ id: 'x' }], backfill_receipts: [{}], bridges: [] }]) assert.equal(typeof m.formatContext(p), 'string');
});
