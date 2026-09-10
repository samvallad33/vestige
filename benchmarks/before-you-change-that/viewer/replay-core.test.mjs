import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {
  ARMS,
  assertPublicEvidence,
  changeTimes,
  finalFacts,
  maxEventOffset,
  parseJsonl,
  resultReports,
  runStatus,
  timelineRows,
  usageTotals,
  validateArmRows,
} from './replay-core.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const evidence = resolve(here, '../evidence');

async function loadEvidence() {
  const manifestText = await readFile(resolve(evidence, 'manifest.json'), 'utf8');
  const timingText = await readFile(resolve(evidence, 'recording-timing.json'), 'utf8');
  const resultText = await readFile(resolve(evidence, 'result.json'), 'utf8');
  assertPublicEvidence(manifestText, 'manifest.json');
  assertPublicEvidence(timingText, 'recording-timing.json');
  assertPublicEvidence(resultText, 'result.json');
  const events = {};
  for (const arm of ARMS) {
    const text = await readFile(resolve(evidence, `${arm}.jsonl`), 'utf8');
    assertPublicEvidence(text, `${arm}.jsonl`);
    events[arm] = parseJsonl(text, `${arm}.jsonl`);
    validateArmRows(arm, events[arm]);
  }
  return {
    manifest: JSON.parse(manifestText),
    recordingTiming: JSON.parse(timingText),
    result: JSON.parse(resultText),
    events,
  };
}

test('public evidence is complete, ordered, and path-sanitized', async () => {
  const {manifest, events} = await loadEvidence();
  assert.deepEqual(manifest.arms, ARMS);
  assert.deepEqual(Object.fromEntries(ARMS.map((arm) => [arm, events[arm].length])), {
    control: 105,
    'mcp-memory-service': 142,
    vestige: 97,
  });
});

test('timeline preserves event order and all recorded waiting intervals', async () => {
  const {manifest, recordingTiming, events} = await loadEvidence();
  assert.equal(recordingTiming.capture_context_utc, '2026-09-07T03:34:04.107Z');
  assert.equal(recordingTiming.dispatch_utc, '2026-09-07T03:34:06.277Z');
  assert.equal(recordingTiming.replay_lead_seconds, 2.17);
  const projected = timelineRows(events, manifest, recordingTiming);
  for (const arm of ARMS) {
    assert.deepEqual(projected[arm].map((row) => row.seq), events[arm].map((row) => row.seq));
    for (let index = 2; index < projected[arm].length; index += 1) {
      assert.ok(projected[arm][index].replay_offset_seconds >= projected[arm][index - 1].replay_offset_seconds);
    }
  }
  assert.ok(maxEventOffset(projected) > 900, 'the recorded 900-second wait remains in the timeline');
  const firstDispatchEvent = projected.control.find((row) => row.kind !== 'task');
  assert.ok(Math.abs(firstDispatchEvent.replay_offset_seconds - 3.569) < 0.00001);
  const times = changeTimes(projected, ['control', 'vestige']);
  assert.equal(times[0], 0);
  assert.ok(times.every((time, index) => index === 0 || time > times[index - 1]));
});

test('result-derived statuses, scores, counters, and lifecycle stay claim-bounded', async () => {
  const {result} = await loadEvidence();
  const reports = resultReports(result);
  assert.equal(runStatus(reports.control), 'Natural completion');
  assert.equal(runStatus(reports.vestige), 'Natural completion');
  assert.equal(runStatus(reports['mcp-memory-service']), 'Timed out at 900s');
  assert.deepEqual(usageTotals(reports.control), {input_tokens: 1182794, cached_input_tokens: 1079808, output_tokens: 18739});
  assert.deepEqual(usageTotals(reports.vestige), {input_tokens: 1302101, cached_input_tokens: 1227648, output_tokens: 15208});
  assert.equal(usageTotals(reports['mcp-memory-service']), null);
  const facts = finalFacts(result);
  assert.deepEqual(facts.scores, ['Control 21/21', 'MCP Memory Service 21/21', 'Vestige 21/21']);
  assert.equal(facts.intentionReturned, true);
  assert.equal(facts.coordinatorCompleted, true);
  assert.equal(result.demo_ready, true, 'historical field is preserved only in raw result evidence');
});

test('private absolute paths fail closed', () => {
  const privatePathFixture = `/${'Users'}/example/private.txt`;
  assert.throws(() => assertPublicEvidence(JSON.stringify({path: privatePathFixture}), 'bad.json'), /private absolute path/);
});
