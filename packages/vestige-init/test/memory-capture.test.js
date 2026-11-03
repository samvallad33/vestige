'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { extractFacts, sentences, admissionOf, linksFor, drainQueue } = require('../hooks/memory.cjs');
const attrs = facts => facts.flatMap(fact => fact.attributes || []);
const ents = facts => facts.flatMap(fact => fact.entities || []);

test('abstract subjects and chatter are not facts', () => {
  assert.deepEqual(extractFacts('Success is almost here!'), []);
  assert.deepEqual(extractFacts('Hope is a good thing.'), []);
  assert.deepEqual(extractFacts('Hard work pays off.'), []);
  assert.deepEqual(extractFacts("I'll look into it."), []);
  assert.deepEqual(extractFacts("I haven't decided yet."), []);
  assert.deepEqual(extractFacts('Next Friday works.'), []);
});

test('moves and job changes carry typed keys and slot handles', () => {
  const facts = extractFacts('Now I live in Denver. I switched jobs: I work at Globex now.');
  assert.ok(attrs(facts).includes('preference:residence=Denver'));
  assert.ok(attrs(facts).includes('preference:employer=Globex'));
  assert.ok(facts.some(fact => fact.slotKey === 'slot:self:residence' && fact.change));
});

test('multi-line prompts split into sentences', () => {
  assert.ok(sentences('I live in Boston.\nI love coffee!').length >= 2);
  assert.ok(extractFacts('I live in Boston.\nI love coffee!').length >= 2);
});

test('each fact has exactly one person entity', () => {
  for (const fact of extractFacts('I live in Boston and I work at Acme as a nurse. My sister loves hiking.')) {
    assert.equal(fact.entities.filter(entity => entity.startsWith('person:')).length, 1);
  }
});

test('pronoun carry-over attaches to the last named person', () => {
  const facts = extractFacts('Maria got married last year. She lives in Lisbon now.');
  assert.ok(facts.length >= 2, JSON.stringify(facts));
  assert.ok(facts.every(fact => fact.entities.includes('person:Maria')));
});

test('word names need a person predicate at sentence start', () => {
  assert.deepEqual(extractFacts('Will is important here.'), []);
});

test('health, diet, medication, allergy attributes', () => {
  const found = attrs(extractFacts('I am vegetarian and I take insulin for my diabetes. I am allergic to peanuts.'));
  for (const key of ['diet:vegetarian', 'medication:insulin', 'condition:diabetes', 'allergy:peanuts']) {
    assert.ok(found.includes(key), `${key} missing from ${found}`);
  }
});

test('preferences carry like and dislike values', () => {
  assert.ok(attrs(extractFacts('I love coffee.')).includes('preference:coffee=like'));
  assert.ok(attrs(extractFacts('I no longer like coffee.')).includes('preference:coffee=dislike'));
});

test('a quoted speaker replaces self', () => {
  assert.ok(ents(extractFacts('Ana said, "I moved to Paris."')).includes('person:Ana'));
});

test('you-statements about the listener are captured, and assistant you is the user', () => {
  const listener = extractFacts('You live in Seattle.');
  assert.ok(ents(listener).includes('person:listener'), JSON.stringify(listener));
  assert.ok(attrs(listener).includes('preference:residence=Seattle'));
  const assistant = extractFacts('You moved to Austin.', {}, { role: 'assistant' });
  assert.ok(ents(assistant).includes('person:self'), JSON.stringify(assistant));
});

test('a residence change links supersedes to the previous slot', () => {
  const next = extractFacts('I moved to Denver.')[0];
  assert.equal(next.slotKey, 'slot:self:residence');
  assert.deepEqual(linksFor(next, { 'slot:self:residence': { id: 'mem-old', content: 'I live in Boston.' } }), [
    { kind: 'supersedes', to: 'mem-old' },
  ]);
  assert.deepEqual(linksFor(next, { 'slot:self:residence': { id: 'mem-old', content: next.content } }), []);
});

test('replay admits, preflight quarantines, and one failure does not abort the batch', async () => {
  const facts = [
    { file: 'a.json', fact: { content: 'first' } },
    { file: 'b.json', fact: { content: 'second' } },
    { file: 'c.json', fact: { content: 'third' } },
    { file: 'd.json', fact: { content: 'fourth' } },
  ];
  let calls = 0;
  const outcome = await drainQueue(facts, async () => {
    calls += 1;
    if (calls === 1) return { success: true, decision: 'replay', replayOf: 'mem-replay' };
    if (calls === 2) return { success: true, decision: 'preflight', wouldWrite: false };
    if (calls === 3) throw new Error('statement admission failed');
    return { success: true, decision: 'create', nodeId: 'mem-new' };
  });
  assert.deepEqual(outcome.saved.map(row => row.id), ['mem-replay', 'mem-new']);
  assert.deepEqual(outcome.quarantined.map(row => row.file), ['b.json', 'c.json']);
  assert.equal(outcome.deferred.length, 0);
  assert.equal(calls, 4, 'the batch kept going after replay and after the throw');
  assert.equal(admissionOf({ decision: 'replay', replayOf: 'mem-replay' }).id, 'mem-replay');
  assert.equal(admissionOf({ decision: 'preflight', wouldWrite: false }).action, 'quarantine');
  assert.equal(admissionOf({ nodeId: 'mem-1' }).action, 'admit');
  assert.equal(admissionOf({ echoNodeId: 'echo-1' }).id, 'echo-1');
});

test('a deadline stops the batch and leaves the rest queued', async () => {
  const facts = [{ file: 'a.json', fact: {} }, { file: 'b.json', fact: {} }];
  const outcome = await drainQueue(facts, async () => {
    const error = new Error('memory deadline exceeded (tools/call)');
    error.deadline = true;
    throw error;
  });
  assert.equal(outcome.saved.length, 0);
  assert.equal(outcome.quarantined.length, 0);
  assert.equal(outcome.deferred.length, 2);
});
