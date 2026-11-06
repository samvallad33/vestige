'use strict';
// Pins the exact hook payloads that crates/vestige-mcp/tests/rsb_hook_capture_stdio.rs
// replays against the real binary, so RSB coverage in CI tracks memory.cjs.
const test = require('node:test');
const assert = require('node:assert/strict');
const m = require('../hooks/memory.cjs');
const item = text => m.ingestItem(m.extractFacts(text)[0]);
const BASE = { node_type: 'fact', source: 'memory-hook:verbatim-user-statement' };
const PINNED = {
  'Maya felt off after dinner at Bangkok Garden': { entities: ['person:Maya', 'place:Bangkok Garden'], attributes: ['health:off'], tags: ['memory-hook-utterance', 'slot:health', 'situation:food-meal', 'situation:outdoors', 'facts-of:Maya'] },
  'Maya has a peanut allergy': { entities: ['person:Maya'], attributes: ['allergy:peanut'], tags: ['memory-hook-utterance', 'slot:health', 'situation:health', 'facts-of:Maya', 'avoid:peanut'] },
  'Maya loves spicy food': { entities: ['person:Maya'], attributes: ['preference:spicy food=like'], tags: ['memory-hook-utterance', 'slot:preference', 'facts-of:Maya'] },
  'Maya hates spicy food': { entities: ['person:Maya'], attributes: ['preference:spicy food=dislike'], tags: ['memory-hook-utterance', 'slot:preference', 'facts-of:Maya'] },
  'Maya lives in Austin': { entities: ['person:Maya', 'place:Austin'], attributes: ['preference:residence=Austin'], tags: ['memory-hook-utterance', 'slot:location', 'slot:maya:residence', 'facts-of:Maya'] },
  'Maya moved to Denver': { entities: ['person:Maya', 'place:Denver'], attributes: ['preference:residence=Denver'], tags: ['memory-hook-utterance', 'slot:location', 'slot:maya:residence', 'facts-of:Maya'] },
  "Noah can't have hot tubs.": { entities: ['person:Noah'], attributes: [], tags: ['memory-hook-utterance', 'slot:avoid', 'facts-of:Noah', 'avoid:hot-tub'] },
  "Mia's birthday is March 3": { entities: ['person:Mia'], attributes: ['preference:birthday=03-03'], tags: ['memory-hook-utterance', 'slot:date', 'slot:mia:birthday', 'facts-of:Mia', 'yearly:03-03'] },
};
test('RSB: hook payloads are exactly what the Rust Backfill acceptance replays', () => {
  for (const [text, want] of Object.entries(PINNED)) assert.deepEqual(item(text), { content: text, ...want, ...BASE }, text);
});
test('RSB: every personal capture carries exactly one person: identity across tags and entities', () => {
  for (const text of [...Object.keys(PINNED), 'I am allergic to shellfish', 'My sister is vegan', 'Maya now prefers loud restaurants', 'Ana said, "I moved to Paris."']) {
    for (const fact of m.extractFacts(text)) {
      const people = [...fact.entities, ...fact.tags, ...fact.attributes].filter(tag => tag.startsWith('person:'));
      assert.equal(people.length, 1, `${text}: ${JSON.stringify(fact)}`);
    }
  }
});
test('RSB: preferences are key=value so a newer value conflicts with the old one', () => {
  for (const text of ['Maya loves spicy food', 'Maya moved to Denver', "Mia's birthday is March 3", 'Maya now prefers loud restaurants']) {
    const prefs = m.extractFacts(text)[0].attributes.filter(tag => tag.startsWith('preference:'));
    assert.ok(prefs.length && prefs.every(tag => /^preference:[^=]+=[^=]+$/.test(tag)), `${text}: ${prefs}`);
  }
  const fact = m.extractFacts('Maya moved to Denver')[0];
  assert.deepEqual(m.linksFor(fact, { [fact.slotKey]: { id: 'mem-old', content: 'Maya lives in Austin' } }), [{ kind: 'supersedes', to: 'mem-old' }]);
});
test('0.3.7: thing words fold case and plurals, keys are 1-3 word runs', () => {
  assert.deepEqual(m.thingWords('Hot Tubs, Strawberries, Tomatoes, glass, cactus'), ['hot', 'tub', 'strawberry', 'tomato', 'glass', 'cactus']);
  assert.equal(m.thingKey('Hot Tubs'), 'hot-tub');
  assert.equal(m.thingKey('a'), null);
  assert.deepEqual(m.thingKeys('big hot tubs'), ['big', 'hot', 'tub', 'big-hot', 'hot-tub', 'big-hot-tub']);
});
test('0.3.7: dates validate and become day:/yearly: only where they belong', () => {
  assert.equal(m.dayOfText('2026-11-03'), '2026-11-03');
  assert.equal(m.dayOfText('2026-13-03'), null);
  assert.equal(m.yearlyOfText('02-30'), '02-30');
  assert.equal(m.yearlyOfText('00-10'), null);
  assert.deepEqual(m.indexTags('self', 'Our anniversary is June 5, 2019', []), ['day:2019-06-05', 'yearly:06-05']);
  assert.deepEqual(m.indexTags('self', 'We fly out June 5', []), []);
  assert.ok(m.extractFacts('We are flying to Paris on 2026-11-03')[0].tags.includes('day:2026-11-03'));
});
test('0.3.7: a fact that ended or never was gets no fact tag; a question is not a fact', () => {
  for (const text of ['Maya outgrew her peanut allergy', 'Maya is no longer allergic to peanuts', 'Is Maya allergic to peanuts?']) {
    const tags = m.extractFacts(text).flatMap(f => [...f.attributes, ...f.tags]);
    assert.ok(!tags.some(tag => /^(?:allergy|avoid|health):/.test(tag)), `${text}: ${tags}`);
  }
});
test('0.3.7: context keeps the engine order (on-target came-back, then safety-first index joins, then BM25) and hides index tags', () => {
  const out = m.formatContext({
    memories: [
      { id: 'c', content: 'Maya felt off at Bangkok Garden', tags: ['person:Maya', 'day:2026-10-01'] },
      { id: 'b', content: 'Maya has a peanut allergy', tags: ['person:Maya', 'allergy:peanut', 'avoid:peanut'] },
      { id: 'a', content: 'Maya likes jazz', tags: ['person:Maya', 'preference:jazz=like', 'facts-of:Maya'] },
      { id: 'z', content: 'Unrelated raised record', tags: [] },
    ],
    backfill_receipts: [{ receipt_id: 'r', memory_id: 'c', reason: 'x', direction: 'raise' }, { receipt_id: 'q', memory_id: 'z', reason: 'y', direction: 'raise' }],
  });
  const ids = out.split('\n').filter(line => line.startsWith('{"id"')).map(line => JSON.parse(line).id);
  assert.deepEqual(ids, ['c', 'b', 'a', 'z']);
  assert.ok(out.includes('Came back: c'), out);
  assert.ok(!/facts-of:|avoid:|day:|yearly:/.test(out), out);
  assert.ok(!/already stated/.test(out) && /search memory/.test(out), out);
});
test('context stays near the baseline size and never claims to be complete', () => {
  const memories = Array.from({ length: 24 }, (_, i) => ({ id: `m${i}`, content: 'x'.repeat(1500), tags: [] }));
  const out = m.formatContext({ memories });
  assert.ok(out.length <= 7000, String(out.length));
  assert.ok(/^\{"id"/m.test(out) && /remember/.test(out));
});

test('compiled reader evidence replaces raw quotes and preserves controlling successors', () => {
  const context = m.formatContext({
    memories: [{ id: 'old', content: 'Mia lives in Austin', tags: ['person:Mia'] }],
    compile_receipt: { hash: 'synthetic' },
    reader_context: 'H2SK receipt=synthetic\n' +
      JSON.stringify({ id: 'old', text: 'Mia lives in Austin', banner: 'OVERRULED BY current' }) + '\n' +
      JSON.stringify({ id: 'current', text: 'Mia lives in Denver', banner: 'CONTROLLING' }),
  });
  assert.equal(context.match(/Mia lives in Austin/g).length, 1);
  assert.equal(context.match(/Mia lives in Denver/g).length, 1);
  assert.match(context, /CONTROLLING/);
  assert.ok(context.length <= 7000);
});
